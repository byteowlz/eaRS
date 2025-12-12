#!/bin/bash
# Setup script for eaRS dictation on Linux (Wayland/X11)
# This configures the uinput kernel module for virtual keyboard support

set -e

echo "========================================"
echo "eaRS Dictation Setup for Linux"
echo "========================================"
echo ""

# Check if running on Linux
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Error: This script is only for Linux systems."
    exit 1
fi

# Check if running with sudo
if [[ $EUID -eq 0 ]]; then
    echo "Error: Do not run this script with sudo."
    echo "The script will ask for sudo permissions when needed."
    exit 1
fi

echo "This script will:"
echo "  1. Load the uinput kernel module"
echo "  2. Configure uinput to load on boot"
echo "  3. Create udev rule for proper /dev/uinput permissions"
echo "  4. Add your user to the 'input' group"
echo ""
echo "You will need to log out and back in after this completes."
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

echo ""
echo "Step 1: Loading uinput kernel module..."
if lsmod | grep -q uinput; then
    echo "✓ uinput module is already loaded"
else
    echo "  Loading uinput module..."
    sudo modprobe uinput
    echo "✓ uinput module loaded"
fi

echo ""
echo "Step 2: Configuring uinput to load on boot..."
if [[ -f /etc/modules-load.d/uinput.conf ]]; then
    if grep -q "^uinput$" /etc/modules-load.d/uinput.conf; then
        echo "✓ uinput is already configured to load on boot"
    else
        echo "  Adding uinput to modules-load.d..."
        echo "uinput" | sudo tee -a /etc/modules-load.d/uinput.conf > /dev/null
        echo "✓ uinput will load on boot"
    fi
else
    echo "  Creating /etc/modules-load.d/uinput.conf..."
    echo "uinput" | sudo tee /etc/modules-load.d/uinput.conf > /dev/null
    echo "✓ uinput will load on boot"
fi

echo ""
echo "Step 3: Creating udev rule for /dev/uinput permissions..."
UDEV_RULE='KERNEL=="uinput", MODE="0660", GROUP="input", OPTIONS+="static_node=uinput"'
if [[ -f /etc/udev/rules.d/99-uinput.conf ]]; then
    if grep -q "uinput" /etc/udev/rules.d/99-uinput.conf; then
        echo "✓ udev rule already exists"
    else
        echo "  Creating udev rule..."
        echo "$UDEV_RULE" | sudo tee /etc/udev/rules.d/99-uinput.conf > /dev/null
        echo "✓ udev rule created"
    fi
else
    echo "  Creating /etc/udev/rules.d/99-uinput.conf..."
    echo "$UDEV_RULE" | sudo tee /etc/udev/rules.d/99-uinput.conf > /dev/null
    echo "✓ udev rule created"
fi

echo "  Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger
echo "✓ udev rules reloaded"

# Reload uinput module to apply new permissions
if lsmod | grep -q uinput; then
    echo "  Reloading uinput module to apply new permissions..."
    sudo rmmod uinput 2>/dev/null || true
    sudo modprobe uinput
    echo "✓ uinput module reloaded with new permissions"
fi

echo ""
echo "Step 4: Adding user to 'input' group..."
if groups "$USER" | grep -q "\binput\b"; then
    echo "✓ User $USER is already in the 'input' group"
    NEED_LOGOUT=false
else
    echo "  Adding $USER to input group..."
    sudo usermod -a -G input "$USER"
    echo "✓ User $USER added to 'input' group"
    NEED_LOGOUT=true
fi

echo ""
echo "Step 5: Verifying /dev/uinput permissions..."
if [[ -c /dev/uinput ]]; then
    PERMS=$(ls -l /dev/uinput)
    echo "  Current permissions: $PERMS"
    
    # Check the group
    UINPUT_GROUP=$(stat -c '%G' /dev/uinput)
    if [[ "$UINPUT_GROUP" == "input" ]]; then
        echo "✓ /dev/uinput has correct group: input"
    else
        echo "⚠ /dev/uinput has incorrect group: $UINPUT_GROUP (should be input)"
        echo "  The udev rule should fix this on next reboot or module reload"
    fi
    
    # Check if accessible
    if [[ -r /dev/uinput ]] && [[ -w /dev/uinput ]]; then
        echo "✓ /dev/uinput is accessible (you can use dictation now!)"
    else
        if [[ "$NEED_LOGOUT" == true ]]; then
            echo "⚠ /dev/uinput will be accessible after you log out and back in"
        else
            echo "⚠ /dev/uinput is not yet accessible"
            echo "  You may need to log out and back in, or reboot"
        fi
    fi
else
    echo "⚠ /dev/uinput does not exist yet"
    echo "  It should appear after the module is loaded (may require reboot)"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""

if [[ "$NEED_LOGOUT" == true ]]; then
    echo "IMPORTANT: You must log out and log back in for group changes to take effect."
    echo ""
    echo "After logging back in, verify the setup:"
    echo "  1. Check group membership: groups | grep input"
    echo "  2. Check /dev/uinput access: ls -l /dev/uinput"
    echo "  3. Start the server: ears server start"
    echo "  4. Test dictation: ears-dictation"
else
    # Check if we can access /dev/uinput right now
    if [[ -r /dev/uinput ]] && [[ -w /dev/uinput ]]; then
        echo "✓ Setup complete! You can use dictation immediately."
        echo ""
        echo "To test:"
        echo "  1. Make sure server is running: ears server start"
        echo "  2. Run dictation client: ears-dictation"
        echo "  3. Open a text editor and speak!"
    else
        echo "You're already in the input group, but /dev/uinput permissions"
        echo "still need to be applied. This might require:"
        echo "  - Logging out and back in"
        echo "  - OR running: sudo rmmod uinput && sudo modprobe uinput"
        echo ""
        echo "After that, test with:"
        echo "  1. Start the server: ears server start"
        echo "  2. Run dictation: ears-dictation"
    fi
fi

echo ""
echo "For more information, see:"
echo "  docs/WAYLAND_VIRTUAL_KEYBOARD.md"
echo ""
