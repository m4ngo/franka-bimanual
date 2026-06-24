temp="$1"

VALID_NAMES=("mario" "luigi")
is_valid_enum() {
    local value="$1"
    for item in "${VALID_NAMES[@]}"; do
        if [[ "$item" == "$value" ]]; then
            return 0
        fi
    done
    return 1
}

if is_valid_enum "$temp"; then
    if [ "$temp" = "mario" ]; then
        sudo ssh -N -L 443:192.168.201.10:443 -i /home/qirico/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new mario@192.168.3.10
    else
        sudo ssh -N -L 443:192.168.200.2:443 -i /home/qirico/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new luigi@192.168.3.11
    fi
else
    echo "Invalid choice. Should either be mario or luigi"
    exit 1
fi