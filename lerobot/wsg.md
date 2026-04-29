2 Basic Command Set 
This chapter describes the basic command set that can be used to grip parts. For an extended set of 
commands providing more functionality, please refer to chapter 3. 
2.1 Interface Control 
2.1.1 Enable verbose mode – VERBOSE 
Enables the interface’s verbose mode. By default, verbose mode is turned off, meaning that the 
module only returns a numeral error code with its error messages. By enabling verbose mode, the 
module additionally returns a text string describing the error. 
Syntax 
VERBOSE=<integer> 
Parameters 
Integer value telling whether verbose mode should be enabled (1) or disabled (0). 
Return String 
VERBOSE=<integer> 
e. g. VERBOSE=0 
2.1.2 Disconnect from interface – BYE 
To disconnect safely from the module, the termination of the connection must be announced before 
closing. 
 If the disconnect is not announced before closing the connection, a FAST STOP will be 
raised blocking all further motion commands. 
Syntax 
BYE() 
Parameters 
No parameters. 
Return String 
ACK BYE  
to acknowledge the command - 9 - 
2.2 
Motion Control 
2.2.1 Referencing the module – HOME 
Execute a homing sequence to reference the gripper fingers. If no parameter is given, referencing will 
be done in default direction. 
This command has to be executed prior to any other motion-related command. The direction of 
homing can be either explicitly specified or can be obtained from the gripper’s configuration. During 
homing, the gripper moves its fingers into the specified direction until it reaches its mechanical end 
stop. The blocking position is used as new origin for all motion-related commands.  
 The best positioning performance will be achieved if homing is done into the direction 
you require the better positioning accuracy. 
 During homing soft limits are disabled! 
 Obstacles in the movement range of the fingers and collision with these during homing 
may result in a wrong reference point for the finger position! 
Syntax 
HOME() 
HOME( <bool> ) 
Parameters 
<bool> (optional) 
Return String 
ACK HOME 
FIN HOME 
Direction of referencing. If 1, referencing will be done in positive direction, if 
0 in negative direction 
to immediately acknowledge command execution 
to indicate command has completed successfully 
2.2.2 Move the gripper fingers – MOVE 
The MOVE command is intended to position the gripper jaws between the gripping cycles, e.g. to 
move the jaws quickly before softly gripping sensitive parts. 
The command expects one or two parameters of which the first one indicates the target position in 
millimeters to which the gripper jaws should be moved and the second parameter indicates a speed 
limit in millimeters per second. 
 Do not use the MOVE command to grip or release parts. Use the GRIP and RELEASE com
mand instead. - 10 - 
 The gripper module has to be homed and must not be in FAST STOP state! 
Syntax 
MOVE( <float> ) 
MOVE( <float>, <float> ) 
Parameters 
<float>  
<float>  
Return String 
ACK MOVE 
FIN MOVE 
Position in mm 
Speed limit in mm/s (optional) 
to immediately acknowledge command execution 
to indicate command has completed successfully 
2.2.3 Grip part – GRIP 
Grip a part. The command’s behavior depends on the number of given parameters: 
1. No parameter. Grip inside until a part is detected. Use default force and speed. 
2. One parameter: Force. Grip inside until a part is detected. Use the given force (in N). 
3. Two parameters: Force, Part width. Grip inside or outside (depending on the current position 
and the target position). Expect a part at the given position. Use the given force (in N). 
If the gripper detects a contact outside the part width tolerance, it is interpreted as a colli
sion and an error is returned. 
4. Three parameters: Force, Position, Speed Limit. Like 3 but use the additional speed limit. 
 Case 1 and 2: As no part width is given, the gripper will grip itself if there is no part be
tween the fingers and the part detection will always set the gripper state to HOLDING. 
You might check the position of the gripper jaws after gripping to make sure a part has 
been gripped. 
If a part width is passed to the GRIP command (i.e. case 3 or 4 in the list above) and the gripper can 
establish the desired force within the defined clamping travel, the gripper state is set to HOLDING 
(part detection feature) and Grip Monitoring will be enabled, i.e. force and position will be continu
ously checked.  
If there was no part between the gripper fingers so they can fall through the clamping travel without 
establishing the full force, the gripper reports that no part was found.  
 The clamping travel and the part width tolerance can be set using the WSG’s web inter
face. Please see the User's Manual for a detailed description of these parameters. - 11 - 
When gripping, the gripper state is updated with the result (either HOLDING or NO PART) as well as 
the gripper statistics (see chapter 3.3.2). If no part was found, the command returns an 
E_CMD_FAILED error.  
Part Detection Feature
If a part width was passed to the GRIP command (i.e. case 3 or 4), the gripper expects a part to be 
found around this position, see the figure below.  
Gripper Module
Positive motion direction
PART
0
Part width tolerance / 2
Clamping travel /2
Nominal part width
Figure 5: Part width tolerance and clamping travel 
If the gripper detects a contact before reaching the part width tolerance area, this is interpreted as a 
collision and an E_AXIS_BLOCKED error is returned. 
 You may reduce the grasping speed with sensitive parts to limit the impact due to the 
mass of the gripper fingers and the internal mechanics. 
 The gripper state reflects the current state of the process. You can read it using the 
GRIPSTATE command (see chapter 2.3.4). 
 It is not possible to send a grip commands while holding a part. In general, a grip com
mand should be always followed by a release command (see chapter 2.2.4) before the 
next grip command is issued. - 12 - 
Syntax 
GRIP() 
GRIP( <float> ) 
GRIP( <float>, <float> ) 
GRIP( <float>, <float>, <float> ) 
Parameters 
<float>  
<float>  
<float>  
Return String 
ACK GRIP 
FIN GRIP 
Force in N (optional) 
Part width in mm (optional) 
Speed limit in mm/s (optional) 
to immediately acknowledge command execution 
to indicate command has completed successfully 
2.2.4 Release part – RELEASE 
Release a previously gripped part. The command’s behavior depends on the number of given param
eters: 
1. No parameter. Open the gripper fingers relative to the current position by the predefined de
fault pull back distance relative to the current position. The default pull back distance can be 
set via the module’s web interface by choosing “Settings” -> “Motion Configuration” from 
the menu. 
2. One parameter: Pull back distance. Open the gripper fingers by the given pull back distance 
relative to the current position. 
3. Two parameters: Pull back distance, speed limit. Open the gripper fingers by the given pull 
back distance relative to the current position using the given speed limit. 
 Release commands are only allowed if the gripper has gripped a part before using the 
GRIP() command. - 13 - 
Gripper Module
Syntax 
RELEASE() 
RELEASE( <float> ) 
RELEASE( <float>, <float> ) 
Parameters 
<float>  
PART
Pull Back
Distance
Figure 6: Pull back distance on release 
Pull back distance in mm (optional) 
<float>  
Return String 
Speed limit in mm/s (optional) 
ACK RELEASE to immediately acknowledge command execution 
FIN RELEASE 
to indicate the command has completed successfully 
2.2.5 Get or set part width tolerance – PWT 
During the execution of a grip command (cf. chapter 2.2.3), the part width tolerance indicates the 
distance before reaching the nominal part width, within which a part is considered to be gripped 
correctly (cf. Figure 5). If the fingers are blocked outside this distance, the grip command returns an 
error. 
The part width tolerance can be set globally using the gripper’s web interface. The command de
scribed here can be used to override the preconfigured value, for example to dynamically adjust the 
part width tolerance to different parts. - 14 - 
 The part width tolerance is changed only for the time the connection is active and the 
changed value only takes effect for grip commands sent via GCL. As soon as the connec
tion is closed, the part width tolerance will be reset to the preconfigured value. 
Syntax 
PWT? 
PWT=<float> 
Parameters 
<float>  
Return string 
PWT=<float> 
Part width tolerance in mm 
2.2.6 Get or set clamping travel – CLT 
During the execution of a grip command (cf. chapter 2.2.3), the clamping travel indicates the distance 
the fingers are allowed to move further after touching a part to apply the desired gripping force (cf. 
Figure 5). If the gripping force can’t be applied within this distance, the grip command returns an 
error. 
At the same time, the clamping travel indicates how far the fingers are allowed to move beyond the 
nominal part width to detect a part. 
If a part is detected before reaching the nominal part width, the clamping travel is measured from 
that point.  If the nominal part width is reached without detecting a part, the clamping travel is 
measured from the nominal part width. 
The clamping travel can be set globally using the gripper’s web interface. The command described 
here can be used to override the preconfigured value, for example to dynamically adjust the clamp
ing travel to different parts. 
 The clamping travel is changed only for the time the connection is active and the changed 
value only takes effect for grip commands sent via GCL. As soon as the connection is 
closed, the clamping travel will be reset to the preconfigured value. 
Syntax 
CLT? 
CLT=<float> 
Parameters 
<float>  
Return string 
CLT=<float> 
Clamping travel in mm - 15 - 
2.3 
Gripper State 
2.3.1 Query current position of gripper jaws – POS 
Returns the current position of the gripper jaws (open width). 
Syntax 
POS? 
Return String 
POS=<float> 
e. g. POS=20.0 
2.3.2 Query current speed of gripper jaws – SPEED 
Get current speed in mm/s. 
Syntax 
SPEED? 
Return String 
SPEED=<float> 
e. g. SPEED=142.0 
2.3.3 Query current gripping force – FORCE 
Get current force value in N. 
Syntax 
FORCE? 
Return String 
FORCE=<float> 
e. g. FORCE=23.0 
2.3.4 Query gripper state – GRIPSTATE 
Get current gripper state. The command returns a numeral value indicating the gripper state. - 16 - 
 See Appendix DFehler! Verweisquelle konnte nicht gefunden werden. for a description of 
he gripper states. 
Syntax 
GRIPSTATE? 
Return String 
GRIPSTATE=<integer> 
e. g. GRIPSTATE=4 to indicate HOLDING. - 17 - 