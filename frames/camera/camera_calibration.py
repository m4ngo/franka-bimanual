"""
Camera calibration with robust, decoupled intrinsics + extrinsics,
using a ChArUco board (checkerboard + embedded ArUco markers).

Workflow
--------
1. INTRINSICS: computed once from many images of the ChArUco board in
   varied poses (glob: .../calibrate/*.png). Bad frames (high
   reprojection error) are filtered out automatically.

2. EXTRINSICS: computed separately via cv.solvePnP from one or more
   *dedicated* images where the board is placed with its CENTER at the
   world origin (glob: .../extrinsics/*.png). If you supply several
   images of this same placement, the resulting poses are averaged
   (translation arithmetically, rotation via quaternion averaging) to
   cancel out per-shot corner-detection noise.

   World axes: the board's own axis convention (origin at the (0,0)
   corner, X along columns, Y along rows, Z out of the board face) is
   NOT necessarily your desired world frame. Set BOARD_TO_WORLD_R (or
   build it with axis_remap_matrix(...) below) to rotate the recentered
   board points into your actual world axes before solvePnP runs. The
   debug image for each extrinsics shot draws the resulting world X/Y/Z
   axes on top of the frame so you can visually confirm they point
   where you expect.

3. CANONICALIZATION IS AUTOMATIC. Unlike a plain checkerboard, a ChArUco
   board's squares/corners are individually identified by the unique
   ArUco marker IDs baked into the board. `CharucoDetector` always
   returns each detected corner tagged with its absolute index in the
   board's own coordinate system (see board.getChessboardCorners()), so
   there is no "which physical corner did detection start from"
   ambiguity to resolve, and no interactive click-to-mark step is
   needed at all -- the marker IDs ARE the canonicalization.

Board specification (fill these in for YOUR physical board):
    SQUARES_X, SQUARES_Y : total squares (black+white) along each edge,
                            i.e. one more than old "internal corners".
    SQUARE_LENGTH        : physical size of one checkerboard square (m).
    MARKER_LENGTH         : physical size of the marker's black-bordered
                            footprint (m) -- NOT the outer white border,
                            NOT the inner bit grid. On this board:
                              inner 4x4 bit grid   = 2.6 cm
                              + black border       = 3.9 cm  <- this one
                              + outer white margin  = 5.0 cm
    ARUCO_DICT            : set to a specific cv.aruco.DICT_* if known;
                            leave as AUTO_DETECT_DICT to let the script
                            try common dictionaries against your images
                            and pick whichever detects the most markers.
"""

import argparse
import glob
import os
import sys

import cv2 as cv
import franka_config as fc
import numpy as np

# ----------------------------------------------------------------------
# Config — every value below comes from config/calibration.yaml.
# `CAM_NAME` selects which camera in config/cameras.yaml is being calibrated;
# override it with `--cam cam_6` on the command line.
# ----------------------------------------------------------------------
_BOARD = fc.calibration("board")
_FIT = fc.calibration("fit")
_PATHS = fc.calibration("paths")
_WORLD_AXES = fc.calibration("world_axes")
SQUARES_X = _BOARD["squares_x"]              # total squares (black+white), horizontal
SQUARES_Y = _BOARD["squares_y"]              # total squares (black+white), vertical
SQUARE_LENGTH = _BOARD["square_length_m"]    # meters, checkerboard square edge
MARKER_LENGTH = _BOARD["marker_length_m"]    # meters, marker's black-bordered footprint

# Set to a specific dict, e.g. cv.aruco.DICT_5X5_250, if you know it.
# Otherwise leave as None and AUTO_DETECT_DICT will probe candidates.
ARUCO_DICT = _BOARD["aruco_dict"]
AUTO_DETECT_DICT = _BOARD["auto_detect_dict"]
CANDIDATE_DICTS = list(_BOARD["candidate_dicts"])

CAM_NAME = "cam_2"


def _apply_cam(cam_name):
    """Point the globs at one camera's frame directories."""
    global CAM_NAME, INTRINSICS_GLOB, EXTRINSICS_GLOB, UNDISTORT_GLOB, UNDISTORT_OUT, MATRICES_OUT
    CAM_NAME = cam_name
    root = str(fc.repo_root() / _PATHS["root"])
    fmt = dict(root=root, cam=cam_name)
    INTRINSICS_GLOB = _PATHS["intrinsics_glob"].format(**fmt)
    EXTRINSICS_GLOB = _PATHS["extrinsics_glob"].format(**fmt)
    UNDISTORT_GLOB = _PATHS["undistort_glob"].format(**fmt)
    UNDISTORT_OUT = _PATHS["undistort_out"].format(**fmt)
    MATRICES_OUT = _PATHS["matrices_out"].format(**fmt)


_apply_cam(CAM_NAME)

REPROJ_ERROR_THRESHOLD = _FIT["reproj_error_threshold_px"]  # px; worse frames are dropped
SHOW_DEBUG_IMAGES = _FIT["show_debug_images"]               # False skips cv.imshow popups
MIN_CHARUCO_CORNERS = _FIT["min_charuco_corners"]           # min corners to accept a frame

CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ----------------------------------------------------------------------
# World axes for extrinsics
# ----------------------------------------------------------------------
# The board's NATIVE frame (as returned by board.getChessboardCorners())
# has its origin at the (0,0) grid corner, with:
#   X -> increases along columns (rightward as printed)
#   Y -> increases along rows (downward as printed)
#   Z -> completes a right-handed system, pointing out of the board face
#
# calibrate_extrinsics() first recenters these points so the origin is
# the board's physical CENTER, then rotates them by BOARD_TO_WORLD_R to
# land in your desired world axes:
#
#     world_pt = BOARD_TO_WORLD_R @ (board_pt - board_center)
#
# Easiest way to set this: use axis_remap_matrix() below, naming which
# (possibly negated) board axis becomes each world axis. Example -- if
# your world X should point along the board's row direction, world Y
# along the negative column direction, and world Z stays the board's Z:
#
#     BOARD_TO_WORLD_R = axis_remap_matrix('Y', '-X', 'Z')
#
# Default is identity: world axes == board's native axes (just
# recentered to the board's middle, no rotation).
def axis_remap_matrix(world_x, world_y, world_z):
    """Build a rotation matrix from a simple axis-remap spec.

    Each of world_x/world_y/world_z is a string naming which BOARD axis
    (optionally prefixed with '-' to negate) becomes that WORLD axis.
    Valid axis names: 'X', 'Y', 'Z'.

    Example: axis_remap_matrix('Y', '-X', 'Z') means
        world X = board's +Y
        world Y = board's -X
        world Z = board's +Z

    Raises if the resulting matrix isn't a valid right-handed rotation
    (e.g. a repeated axis, or an odd number of sign flips that would
    mirror the frame instead of rotating it).
    """
    axis_map = {
        'X': np.array([0.0, 1.0, 0.0]),
        'Y': np.array([1.0, 0.0, 0.0]),
        'Z': np.array([0.0, 0.0, -1.0]),
    }

    def parse(spec):
        spec = spec.strip()
        sign = -1.0 if spec.startswith('-') else 1.0
        name = spec.lstrip('+-').upper()
        if name not in axis_map:
            raise ValueError(f"invalid axis spec '{spec}' -- use X, Y, or Z (optionally '-' prefixed)")
        return sign * axis_map[name]

    R = np.array([parse(world_x), parse(world_y), parse(world_z)], dtype=np.float64)

    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        raise ValueError(
            "axis remap is not orthonormal -- each world axis must map to a "
            "distinct board axis (check for repeats)"
        )
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(
            f"axis remap has determinant {det:.3f}, expected +1 -- this would "
            "mirror the frame instead of rotating it. Flip the sign on exactly "
            "one axis to fix handedness."
        )
    return R


# From config/calibration.yaml (world_axes.board_to_world).
BOARD_TO_WORLD_R = axis_remap_matrix(*_WORLD_AXES["board_to_world"])

# This script emits extrinsics in the CALIBRATION frame, whose origin is always
# the board CENTER:
#
#     calib_pt = BOARD_TO_WORLD_R @ (board_pt - board_center)
#
# It deliberately has no origin-offset knob. Where that frame sits in the world
# lives in exactly one place, world.yaml: calib_origin_in_world, and is applied
# at load time by franka_config — so moving the world origin never invalidates
# a calibration. An origin_offset_m left in calibration.yaml is a leftover from
# the old two-knob scheme and is rejected rather than silently ignored.
if "origin_offset_m" in _WORLD_AXES:
    raise SystemExit(
        "config/calibration.yaml still has world_axes.origin_offset_m. That knob was "
        "folded into world.yaml: calib_origin_in_world — delete it, and make sure its "
        "value is reflected there (extrinsics are now board-centre relative)."
    )


def draw_world_axes(img, mtx, dist, rvec, tvec, length=0.1):
    """Overlay the world X (red), Y (green), Z (blue) axes at the world
    origin on a debug image, so you can visually confirm BOARD_TO_WORLD_R
    is doing what you expect before trusting the numeric pose."""
    dbg = img.copy()
    cv.drawFrameAxes(dbg, mtx, dist, rvec, tvec, length, 3)
    return dbg


# ----------------------------------------------------------------------
# Board / detector setup
# ----------------------------------------------------------------------
def build_board(dictionary):
    board = cv.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary
    )
    return board


def make_detector(board):
    aruco_params = cv.aruco.DetectorParameters()
    charuco_params = cv.aruco.CharucoParameters()
    detector = cv.aruco.CharucoDetector(board, charuco_params, aruco_params)
    return detector


def autodetect_dictionary(sample_images):
    """Try each candidate ArUco dictionary against a handful of sample
    images and return the one that detects the most markers overall."""
    print("[dict] ARUCO_DICT not set -- probing candidate dictionaries...")
    best_name, best_count = None, -1

    for name in CANDIDATE_DICTS:
        dict_id = getattr(cv.aruco, name)
        dictionary = cv.aruco.getPredefinedDictionary(dict_id)
        aruco_params = cv.aruco.DetectorParameters()
        aruco_detector = cv.aruco.ArucoDetector(dictionary, aruco_params)

        total = 0
        for fname in sample_images:
            img = cv.imread(fname)
            if img is None:
                continue
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            corners, ids, _ = aruco_detector.detectMarkers(gray)
            if ids is not None:
                total += len(ids)

        print(f"    {name:16s} total markers detected: {total}")
        if total > best_count:
            best_count = total
            best_name = name

    if best_count <= 0:
        raise RuntimeError(
            "No ArUco markers detected with any candidate dictionary. "
            "Check image paths, lighting, or set ARUCO_DICT explicitly."
        )

    print(f"[dict] selected {best_name} ({best_count} total markers across samples)")
    return getattr(cv.aruco, best_name)


def resolve_dictionary():
    if ARUCO_DICT is not None:
        return cv.aruco.getPredefinedDictionary(ARUCO_DICT)

    if not AUTO_DETECT_DICT:
        raise RuntimeError("ARUCO_DICT is None and AUTO_DETECT_DICT is False; set one of them.")

    sample_images = glob.glob(INTRINSICS_GLOB)[:8] or glob.glob(EXTRINSICS_GLOB)[:8]
    if not sample_images:
        raise RuntimeError("No images found to auto-detect the ArUco dictionary from.")

    dict_id = autodetect_dictionary(sample_images)
    return cv.aruco.getPredefinedDictionary(dict_id)


def normalize_charuco(corners, ids):
    """OpenCV 5's detectBoard returns (N,2)/(N,); the aruco helpers need (N,1,2)/(N,1)."""
    if corners is None or ids is None:
        return corners, ids
    return (np.asarray(corners, np.float32).reshape(-1, 1, 2),
            np.asarray(ids, np.int32).reshape(-1, 1))


# ----------------------------------------------------------------------
# calibrateCameraCharuco replacement (removed in newer OpenCV; the
# documented replacement is CharucoBoard.matchImagePoints + calibrateCamera)
# ----------------------------------------------------------------------
def calibrate_camera_charuco(all_charuco_corners, all_charuco_ids, board, img_shape):
    """Drop-in replacement for the removed cv.aruco.calibrateCameraCharuco.
    Uses board.matchImagePoints() to turn each frame's (charuco_corners,
    charuco_ids) into matched (objPoints, imgPoints), then feeds the
    per-frame point lists straight into cv.calibrateCamera()."""
    all_obj_points = []
    all_img_points = []
    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < 4:
            continue
        all_obj_points.append(obj_pts)
        all_img_points.append(img_pts)

    if len(all_obj_points) < 3:
        raise RuntimeError("Not enough matchable frames to calibrate (need >= 3).")

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        all_obj_points, all_img_points, img_shape, None, None
    )
    return ret, mtx, dist, rvecs, tvecs


# ----------------------------------------------------------------------
# Stage 1: Intrinsics
# ----------------------------------------------------------------------
def calibrate_intrinsics(detector, board):
    images = glob.glob(INTRINSICS_GLOB)
    print(f"[intrinsics] found {len(images)} images")

    all_charuco_corners, all_charuco_ids, fnames_all = [], [], []
    img_shape = None

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        charuco_corners, charuco_ids = normalize_charuco(charuco_corners, charuco_ids)

        if charuco_corners is None or len(charuco_corners) < MIN_CHARUCO_CORNERS:
            n = 0 if charuco_corners is None else len(charuco_corners)
            print(f"  [skip] only {n} charuco corners in {fname}")
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        fnames_all.append(fname)

        if SHOW_DEBUG_IMAGES:
            dbg = img.copy()
            if marker_ids is not None:
                cv.aruco.drawDetectedMarkers(dbg, marker_corners, marker_ids)
            cv.aruco.drawDetectedCornersCharuco(dbg, charuco_corners, charuco_ids)
            cv.imshow('intrinsics: detected charuco corners', dbg)
            cv.waitKey(200)

    if SHOW_DEBUG_IMAGES:
        cv.destroyAllWindows()

    if len(all_charuco_corners) < 3:
        raise RuntimeError("Not enough valid intrinsics images (need >= 3, ideally 15-20+).")

    ret, mtx, dist, rvecs, tvecs = calibrate_camera_charuco(
        all_charuco_corners, all_charuco_ids, board, img_shape
    )

    # Per-image reprojection error -> filter outliers, then refit
    board_corners_3d = board.getChessboardCorners()
    per_image_error = []
    for i in range(len(all_charuco_corners)):
        obj_pts = board_corners_3d[all_charuco_ids[i].flatten()]
        proj, _ = cv.projectPoints(obj_pts, rvecs[i], tvecs[i], mtx, dist)
        # reshape both to flat (N, 2) before cv.norm -- detected corners
        # and projectPoints() output don't always come back with matching
        # channel-count/dtype, which otherwise raises a type-mismatch error.
        detected = all_charuco_corners[i].reshape(-1, 2)
        proj2 = proj.reshape(-1, 2)
        err = cv.norm(detected, proj2, cv.NORM_L2) / len(proj)
        per_image_error.append(err)

    print("[intrinsics] per-image reprojection error (px):")
    for f, e in zip(fnames_all, per_image_error):
        flag = "  <-- DROPPED (high error)" if e > REPROJ_ERROR_THRESHOLD else ""
        print(f"  {os.path.basename(f):30s} {e:.4f}{flag}")

    keep = [i for i, e in enumerate(per_image_error) if e <= REPROJ_ERROR_THRESHOLD]
    dropped = len(all_charuco_corners) - len(keep)
    if dropped > 0 and len(keep) >= 3:
        print(f"[intrinsics] dropping {dropped} high-error frame(s), refitting...")
        corners_f = [all_charuco_corners[i] for i in keep]
        ids_f = [all_charuco_ids[i] for i in keep]
        ret, mtx, dist, rvecs, tvecs = calibrate_camera_charuco(
            corners_f, ids_f, board, img_shape
        )
        mean_error = 0
        for i in range(len(corners_f)):
            obj_pts = board_corners_3d[ids_f[i].flatten()]
            proj, _ = cv.projectPoints(obj_pts, rvecs[i], tvecs[i], mtx, dist)
            detected = corners_f[i].reshape(-1, 2)
            proj2 = proj.reshape(-1, 2)
            mean_error += cv.norm(detected, proj2, cv.NORM_L2SQR) / len(proj)
        total_error = np.sqrt(mean_error / len(corners_f))
    else:
        if dropped > 0:
            print(f"[intrinsics] {dropped} frame(s) exceed threshold but too few would remain; keeping all.")
        mean_error = 0
        for i in range(len(all_charuco_corners)):
            obj_pts = board_corners_3d[all_charuco_ids[i].flatten()]
            proj, _ = cv.projectPoints(obj_pts, rvecs[i], tvecs[i], mtx, dist)
            detected = all_charuco_corners[i].reshape(-1, 2)
            proj2 = proj.reshape(-1, 2)
            mean_error += cv.norm(detected, proj2, cv.NORM_L2SQR) / len(proj)
        total_error = np.sqrt(mean_error / len(all_charuco_corners))

    print(f"\n[intrinsics] final RMS reprojection error: {total_error:.4f} px  "
          f"({'good' if total_error < 0.5 else 'check calibration quality'})")
    print("[intrinsics] camera matrix:\n", mtx)
    print("[intrinsics] distortion coeffs:\n", dist.ravel())

    return mtx, dist, img_shape


# ----------------------------------------------------------------------
# Rotation averaging (quaternion-based) for multi-shot extrinsics
# ----------------------------------------------------------------------
def average_rotations(rvecs):
    """Average a list of rotation vectors via quaternion averaging
    (handles sign ambiguity), returns an averaged rotation vector."""
    quats = []
    for rvec in rvecs:
        R, _ = cv.Rodrigues(rvec)
        q = rotmat_to_quaternion(R)
        quats.append(q)
    quats = np.array(quats)

    for i in range(1, len(quats)):
        if np.dot(quats[0], quats[i]) < 0:
            quats[i] = -quats[i]

    M = quats.T @ quats
    eigvals, eigvecs = np.linalg.eigh(M)
    avg_q = eigvecs[:, np.argmax(eigvals)]
    if avg_q[3] < 0:
        avg_q = -avg_q

    R_avg = quaternion_to_rotmat(avg_q)
    rvec_avg, _ = cv.Rodrigues(R_avg)
    return rvec_avg


def rotmat_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([x, y, z, w])


def quaternion_to_rotmat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


# ----------------------------------------------------------------------
# Stage 2: Extrinsics via solvePnP (board center = world origin,
# rotated into custom world axes via BOARD_TO_WORLD_R)
# ----------------------------------------------------------------------
def calibrate_extrinsics(mtx, dist, detector, board):
    images = glob.glob(EXTRINSICS_GLOB)
    print(f"\n[extrinsics] found {len(images)} dedicated origin-pose image(s)")
    if len(images) == 0:
        raise RuntimeError(f"No extrinsics images found at {EXTRINSICS_GLOB}")

    # Board corners are defined by the board itself in its own frame, with
    # origin at one corner of the grid. Shift to board-CENTER origin and rotate
    # into the calibration axes (see BOARD_TO_WORLD_R above). The origin stays
    # the board centre — world placement is world.yaml's job.
    board_corners_3d = board.getChessboardCorners()
    center = board_corners_3d.mean(axis=0)
    board_corners_centered = board_corners_3d - center
    board_corners_world = board_corners_centered @ BOARD_TO_WORLD_R.T

    rvecs_all, tvecs_all = [], []

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        charuco_corners, charuco_ids = normalize_charuco(charuco_corners, charuco_ids)

        if charuco_corners is None or len(charuco_corners) < MIN_CHARUCO_CORNERS:
            n = 0 if charuco_corners is None else len(charuco_corners)
            print(f"  [skip] only {n} charuco corners in {fname}")
            continue

        # No manual canonicalization needed: charuco_ids gives the exact
        # board-frame index of every detected corner, so we look up each
        # corner's 3D position directly instead of assuming a traversal order.
        obj_pts = board_corners_world[charuco_ids.flatten()]

        ok, rvec, tvec = cv.solvePnP(
            obj_pts, charuco_corners, mtx, dist,
            flags=cv.SOLVEPNP_ITERATIVE
        )
        if not ok:
            print(f"  [skip] solvePnP failed for {fname}")
            continue

        rvec, tvec = cv.solvePnPRefineLM(obj_pts, charuco_corners, mtx, dist, rvec, tvec)

        if SHOW_DEBUG_IMAGES:
            dbg = img.copy()
            if marker_ids is not None:
                cv.aruco.drawDetectedMarkers(dbg, marker_corners, marker_ids)
            cv.aruco.drawDetectedCornersCharuco(dbg, charuco_corners, charuco_ids)
            # Draw the WORLD axes (per BOARD_TO_WORLD_R) at the origin so
            # you can visually confirm they point where you expect.
            dbg = draw_world_axes(dbg, mtx, dist, rvec, tvec, length=SQUARE_LENGTH * 2)
            cv.imshow('extrinsics: detected charuco board + world axes', dbg)
            cv.waitKey(0)

        rvecs_all.append(rvec)
        tvecs_all.append(tvec)
        print(f"  [{os.path.basename(fname)}] used {len(charuco_corners)} charuco corners")

    if SHOW_DEBUG_IMAGES:
        cv.destroyAllWindows()

    if len(rvecs_all) == 0:
        raise RuntimeError("No valid extrinsics poses computed.")

    tvec_avg = np.mean(np.array(tvecs_all), axis=0)
    rvec_avg = average_rotations(rvecs_all) if len(rvecs_all) > 1 else rvecs_all[0]

    if len(rvecs_all) > 1:
        t_std = np.std(np.array(tvecs_all).reshape(-1, 3), axis=0)
        print(f"[extrinsics] translation std-dev across {len(rvecs_all)} shots (m): {t_std}")

    R_avg, _ = cv.Rodrigues(rvec_avg)

    print("\n[extrinsics] R (calib->camera):\n", R_avg)
    print("[extrinsics] t (calib->camera):\n", tvec_avg.flatten())

    R_cam_in_world = R_avg.T
    t_cam_in_world = -R_avg.T @ tvec_avg

    print("\n[extrinsics] Camera pose in the CALIBRATION frame, origin at the board")
    print("centre (BOARD_TO_WORLD_R applied). world.yaml: calib_origin_in_world lifts")
    print("this into world at load time.")
    print("R_cam_in_calib:\n", R_cam_in_world)
    print("t_cam_in_calib (camera position, meters):\n", t_cam_in_world.flatten())

    return R_avg, tvec_avg, R_cam_in_world, t_cam_in_world


# ----------------------------------------------------------------------
# Optional: undistort a sample image
# ----------------------------------------------------------------------
def undistort_sample(mtx, dist):
    images = glob.glob(UNDISTORT_GLOB)
    for fname in images:
        img = cv.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w2, h2 = roi
        dst = dst[y:y+h2, x:x+w2]
        os.makedirs(os.path.dirname(UNDISTORT_OUT), exist_ok=True)
        cv.imwrite(UNDISTORT_OUT, dst)
        print(f"[undistort] wrote {UNDISTORT_OUT}")


def emit_cameras_yaml(cam_name, mtx, dist, R_cw, t_cw):
    """Print the config/cameras.yaml calibration block for this camera.

    Extrinsics stay in the CALIBRATION frame (board centre on the table);
    config/world.yaml lifts them into world via calib_origin_in_world, so no
    z-offset is baked in here.
    """
    def rows(m):
        return "\n".join(
            "        - [" + ", ".join(f"{v:.8f}" for v in row) + "]" for row in m
        )

    print(f"\n# --- paste into config/cameras.yaml under cameras.{cam_name} ---")
    print("    calibration:")
    print(f"      source: {_PATHS['root']}/{cam_name}_matrices.txt")
    print("      intrinsic_matrix:")
    print(rows(np.asarray(mtx)))
    print("      distortion_coeffs: [" + ", ".join(f"{v:.8e}" for v in np.asarray(dist).ravel()) + "]")
    print("      r_cam_in_calib:")
    print(rows(np.asarray(R_cw)))
    print("      t_cam_in_calib: [" + ", ".join(f"{v:.8f}" for v in np.asarray(t_cw).ravel()) + "]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChArUco intrinsics + extrinsics calibration")
    parser.add_argument("--cam", default=CAM_NAME,
                        help="camera id from config/cameras.yaml (default: %(default)s)")
    parser.add_argument("--no-debug-images", action="store_true",
                        help="skip cv.imshow popups regardless of config")
    args = parser.parse_args()

    if args.cam not in fc.camera_keys():
        parser.error(f"unknown camera {args.cam!r}; known: {', '.join(fc.camera_keys())}")
    _apply_cam(args.cam)
    if args.no_debug_images:
        SHOW_DEBUG_IMAGES = False

    dictionary = resolve_dictionary()
    board = build_board(dictionary)
    detector = make_detector(board)

    mtx, dist, img_shape = calibrate_intrinsics(detector, board)
    R_wc, t_wc, R_cw, t_cw = calibrate_extrinsics(mtx, dist, detector, board)
    undistort_sample(mtx, dist)
    emit_cameras_yaml(args.cam, mtx, dist, R_cw, t_cw)