#!/usr/bin/env bash
# Per-user app transaction. No TCC, login items, quarantine changes or payload execution.
# App copies/backups are retained; the only removed file is a validated completed journal.
set -euo pipefail
umask 077
export PATH=/usr/bin:/bin:/usr/sbin:/sbin LC_ALL=C
prefix="${HOME:-}/Applications"
state="${XDG_STATE_HOME:-${HOME:-}/.local/state}/ears/install"
source_app="" pin="" action=install seen='|' help=false
usage() {
    printf '%s\n' \
      'install-macos.sh --app PATH --app-cdhash TRUSTED_40_HEX [--prefix PATH] [--state-dir PATH]' \
      'install-macos.sh --rollback|--uninstall|--recover [--prefix PATH] [--state-dir PATH]' \
      'Quit eaRS first. App backups and all external settings/models/history are retained.'
}
parse_error() { printf '%s\n' "$*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
    option=$1
    case "$seen" in *"|$option|"*) parse_error "Duplicate option: $option" ;; esac
    seen="$seen$option|"
    case "$option" in
        --app|--app-cdhash|--prefix|--state-dir)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || parse_error "Missing value: $option"
            case "$option" in
                --app) source_app=$2 ;; --app-cdhash) pin=$2 ;;
                --prefix) prefix=$2 ;; --state-dir) state=$2 ;;
            esac
            shift 2 ;;
        --rollback|--uninstall|--recover)
            [[ "$action" == install ]] || parse_error 'Conflicting actions'
            action=${option#--}; shift ;;
        --help) help=true; shift ;;
        *) parse_error "Unknown option: $option" ;;
    esac
done
if $help; then [[ "$seen" == '|--help|' ]] || parse_error 'Use --help alone'; usage; exit 0; fi
if [[ "$action" == install ]]; then
    [[ -n "$source_app" && "$pin" =~ ^[0-9a-fA-F]{40}$ ]] || parse_error 'Install requires --app and externally trusted --app-cdhash (40 hex)'
    pin=$(printf '%s' "$pin" | /usr/bin/tr A-F a-f)
else
    [[ -z "$source_app$pin" ]] || parse_error '--app/--app-cdhash are install-only'
fi
fail() { printf '%s\n' "$*" >&2; return 1; }
exists() { [[ -e "$1" || -L "$1" ]]; }
hash_ok() { [[ "$1" =~ ^[0-9a-f]{40}$ ]]; }
sha_ok() { [[ "$1" =~ ^[0-9a-f]{64}$ ]]; }
safe_path() {
    [[ "$1" == /* && "$1" != / && ${#1} -le 1024 && "$1" != *[[:cntrl:]]* && "$1" != *'"'* && "$1" != *'\'* ]] || return 1
    case "$1/" in *'//'*|*'/./'*|*'/../'*) return 1 ;; esac
}
# Resolve an existing ancestor, without creating anything. Final symlinks refuse.
canonical_directory() {
    local path=${1%/} tail=''
    safe_path "$path" || { fail 'Unsupported absolute directory path'; return 1; }
    while ! exists "$path"; do
        tail="/${path##*/}$tail"; path=${path%/*}; [[ -n "$path" ]] || path=/
    done
    [[ -d "$path" && ! -L "$path" ]] || { fail 'Directory/symlink collision'; return 1; }
    local base; base=$(cd "$path" && pwd -P) || return 1
    printf '%s%s\n' "${base%/}" "$tail"
}
[[ $(/usr/bin/uname -s) == Darwin ]] || parse_error 'macOS required'
prefix=$(canonical_directory "$prefix") || exit 2
state=$(canonical_directory "$state") || exit 2
case "$prefix/" in "$state/"*) parse_error 'Prefix and state must be disjoint' ;; esac
case "$state/" in "$prefix/"*) parse_error 'Prefix and state must be disjoint' ;; esac
destination="$prefix/eaRS.app" receipt="$state/receipt.json" journal="$state/journal.json"
uid=$(/usr/bin/id -u)
owned_directory() {
    [[ -d "$1" && ! -L "$1" && $(/usr/bin/stat -f %u "$1") == "$uid" ]] || { fail 'Untrusted directory'; return 1; }
    local mode; mode=$(/usr/bin/stat -f %Lp "$1") || return 1
    (( (8#$mode & 0022) == 0 )) || { fail 'Group/world-writable directory'; return 1; }
}
regular() {
    [[ -f "$1" && ! -L "$1" && $(/usr/bin/stat -f %l "$1") == 1 && $(/usr/bin/stat -f %u "$1") == "$uid" ]] || { fail 'Metadata must be an owned regular non-symlink single-link file'; return 1; }
    local size mode; size=$(/usr/bin/stat -f %z "$1") || return 1
    mode=$(/usr/bin/stat -f %Lp "$1") || return 1
    [[ "$size" -le 16384 ]] && (( (8#$mode & 0022) == 0 )) || { fail 'Unsafe metadata size/permissions'; return 1; }
}
sha() { /usr/bin/shasum -a 256 | /usr/bin/awk '{print $1}'; }
file_sha() { regular "$1" || return 1; /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'; }
field() {
    [[ $(/usr/bin/plutil -type "$2" "$1") == string ]] || return 1
    /usr/bin/plutil -extract "$2" raw -o - "$1"
}
tx_path() {
    local suffix
    case "$1" in "$prefix"/.ears-install.*) suffix=${1#"$prefix"/.ears-install.} ;; *) return 1 ;; esac
    [[ "$suffix" =~ ^[a-zA-Z0-9]{6}$ ]] && owned_directory "$1"
}
backup_path() { [[ "$1" == */old.app ]] && tx_path "${1%/old.app}"; }
receipt_json() {
    printf '{"schema":"ears.install/v2","path":"%s","current_cdhash":"%s","previous_path":"%s","previous_cdhash":"%s"}\n' "$destination" "$1" "$2" "$3"
}
read_receipt() {
    local file=$1 schema path expected actual
    regular "$file" || return 1
    schema=$(field "$file" schema) && path=$(field "$file" path) || { fail 'Invalid receipt fields'; return 1; }
    [[ "$schema" == ears.install/v2 && "$path" == "$destination" ]] || { fail 'Unknown/mismatched receipt; no migration or adoption'; return 1; }
    R_current=$(field "$file" current_cdhash) && R_previous=$(field "$file" previous_path) && R_previous_hash=$(field "$file" previous_cdhash) || return 1
    [[ -z "$R_current" ]] || hash_ok "$R_current" || return 1
    if [[ -n "$R_previous" ]]; then
        hash_ok "$R_previous_hash" && backup_path "$R_previous" || { fail 'Unsafe receipt backup'; return 1; }
    else [[ -z "$R_previous_hash" ]] || return 1; fi
    # Exact canonical bytes reject unknown/duplicate keys, alternate JSON types,
    # escaped/control paths, extra whitespace and old schemas (deliberately strict).
    expected=$(receipt_json "$R_current" "$R_previous" "$R_previous_hash" | sha) && actual=$(file_sha "$file") || return 1
    [[ "$expected" == "$actual" ]] || { fail 'Noncanonical/unknown receipt content'; return 1; }
}
verify_app() {
    local app=$1 expected=$2 actual bundle_id
    hash_ok "$expected" || return 1
    [[ -d "$app" && ! -L "$app" && -f "$app/Contents/Info.plist" && ! -L "$app/Contents/Info.plist" ]] || { fail 'Invalid application path'; return 1; }
    bundle_id=$(/usr/libexec/PlistBuddy -c 'Print CFBundleIdentifier' "$app/Contents/Info.plist") || return 1
    [[ "$bundle_id" == dev.byteowlz.ears-ui ]] || { fail 'Not an eaRS bundle'; return 1; }
    /usr/bin/codesign --verify --deep --strict "$app" || return 1
    actual=$(/usr/bin/codesign -d --verbose=4 "$app" 2>&1 | /usr/bin/awk -F= '/^CDHash=/{print $2}') || return 1
    [[ "$actual" == "$expected" ]] || { fail 'Application CDHash does not match trusted pin/receipt'; return 1; }
    /usr/sbin/spctl --assess --type execute "$app" || return 1
}
move_absent() {
    ! exists "$2" || { fail 'Refusing rename over an existing/unmanaged entry'; return 1; }
    /bin/mv -n -- "$1" "$2" || return 1
    ! exists "$1" && exists "$2" || { fail 'Rename did not complete'; return 1; }
}
journal_json() {
    printf '{"schema":"ears.install-journal/v1","path":"%s","transaction":"%s","old_cdhash":"%s","new_cdhash":"%s","old_receipt_sha256":"%s","next_receipt":"%s","next_receipt_sha256":"%s","previous_path":"%s","previous_cdhash":"%s"}\n' \
      "$destination" "$J_tx" "$J_old" "$J_new" "$J_old_receipt" "$J_next" "$J_next_sha" "$J_previous" "$J_previous_hash"
}
read_journal() {
    local schema path suffix expected actual
    regular "$journal" || return 1
    schema=$(field "$journal" schema) && path=$(field "$journal" path) || return 1
    [[ "$schema" == ears.install-journal/v1 && "$path" == "$destination" ]] || { fail 'Unknown/mismatched journal'; return 1; }
    J_tx=$(field "$journal" transaction) && J_old=$(field "$journal" old_cdhash) && J_new=$(field "$journal" new_cdhash) && J_old_receipt=$(field "$journal" old_receipt_sha256) && J_next=$(field "$journal" next_receipt) && J_next_sha=$(field "$journal" next_receipt_sha256) && J_previous=$(field "$journal" previous_path) && J_previous_hash=$(field "$journal" previous_cdhash) || return 1
    tx_path "$J_tx" && sha_ok "$J_next_sha" || return 1
    [[ -z "$J_old_receipt" ]] || sha_ok "$J_old_receipt" || return 1
    [[ -z "$J_old" ]] || hash_ok "$J_old" || return 1
    [[ -z "$J_new" ]] || hash_ok "$J_new" || return 1
    [[ "$J_old" != "$J_new" && ( -n "$J_old_receipt" || -z "$J_old" ) ]] || return 1
    case "$J_next" in "$state"/.ears-receipt.*) suffix=${J_next#"$state"/.ears-receipt.} ;; *) return 1 ;; esac
    [[ "$suffix" =~ ^[a-zA-Z0-9]{6}$ ]] || return 1
    if [[ -n "$J_previous" ]]; then hash_ok "$J_previous_hash" && backup_path "$J_previous" || return 1
    else [[ -z "$J_previous_hash" ]] || return 1; fi
    if [[ -n "$J_old" ]]; then
        [[ "$J_previous" == "$J_tx/old.app" && "$J_previous_hash" == "$J_old" ]] || return 1
    fi
    expected=$(journal_json | sha) && actual=$(file_sha "$journal") || return 1
    [[ "$expected" == "$actual" ]] || { fail 'Noncanonical/unknown journal content'; return 1; }
}
receipt_fingerprint() {
    if exists "$receipt"; then read_receipt "$receipt" && file_sha "$receipt"
    else printf '\n'; fi
}
expect_receipt() {
    local actual; actual=$(receipt_fingerprint) || return 1
    [[ "$actual" == "$1" ]] || { fail 'Receipt changed before commit; refusing overwrite'; return 1; }
}
recover_transaction() {
    exists "$journal" || return 0
    read_journal || { fail 'Recovery refused: inspect retained journal and app copies'; return 1; }
    local fingerprint; fingerprint=$(receipt_fingerprint) || return 1
    if [[ "$fingerprint" == "$J_next_sha" ]]; then
        read_receipt "$receipt" || return 1
        [[ "$R_current" == "$J_new" && "$R_previous" == "$J_previous" && "$R_previous_hash" == "$J_previous_hash" ]] || return 1
        if [[ -n "$J_new" ]]; then verify_app "$destination" "$J_new" || return 1
        else ! exists "$destination" || { fail 'Unexpected destination after uninstall'; return 1; }; fi
        [[ -z "$J_previous" ]] || verify_app "$J_previous" "$J_previous_hash" || return 1
        printf '%s\n' 'Recovered completed receipt commit; backups retained.'
    elif [[ "$fingerprint" == "$J_old_receipt" ]]; then
        [[ $(file_sha "$J_next") == "$J_next_sha" ]] && read_receipt "$J_next" || return 1
        [[ "$R_current" == "$J_new" && "$R_previous" == "$J_previous" && "$R_previous_hash" == "$J_previous_hash" ]] || return 1
        if [[ -n "$fingerprint" ]]; then
            read_receipt "$receipt" || return 1
            [[ "$R_current" == "$J_old" ]] || return 1
            [[ -z "$R_previous" ]] || verify_app "$R_previous" "$R_previous_hash" || return 1
        fi
        if [[ -n "$J_old" ]]; then
            if exists "$J_tx/old.app"; then
                verify_app "$J_tx/old.app" "$J_old" || return 1
                if exists "$destination"; then
                    verify_app "$destination" "$J_new" || return 1
                    move_absent "$destination" "$J_tx/rejected.app" || return 1
                fi
                move_absent "$J_tx/old.app" "$destination" || return 1
            fi
            verify_app "$destination" "$J_old" || return 1
        elif exists "$destination"; then
            verify_app "$destination" "$J_new" || return 1
            move_absent "$destination" "$J_tx/rejected.app" || return 1
        fi
        printf '%s\n' 'Recovered previous receipt selection; uncommitted app copies retained.'
    else
        fail 'Receipt changed outside transaction; recovery refuses to overwrite anything'; return 1
    fi
    # No app/staging/backup recursion. Only this validated, completed WAL is removed.
    regular "$journal" && /bin/rm -- "$journal"
}
# Validate metadata before any directory/lock creation, even when the app is absent.
! exists "$state/lock" || { printf '%s\n' 'Legacy/unknown lock entry; inspect old installer state first' >&2; exit 5; }
if exists "$receipt"; then read_receipt "$receipt" || { printf '%s\n' 'Receipt metadata rejected' >&2; exit 5; }; fi
if [[ "$action" == install ]]; then
    [[ -d "$source_app" && ! -L "$source_app" ]] || { fail 'Invalid/symlink source application'; exit 2; }
    source_app=$(cd "$source_app" && pwd -P)
    safe_path "$source_app" || exit 2
    case "$source_app/" in "$prefix/"*|"$state/"*) fail 'Install source must be outside prefix/state'; exit 2 ;; esac
fi
/bin/mkdir -p "$prefix" "$state"
owned_directory "$prefix" && owned_directory "$state" || exit 5
prepare_lock() {
    if ! exists "$1"; then (set -o noclobber; : > "$1") || return 1; fi
    regular "$1" && [[ $(/usr/bin/stat -f %z "$1") == 0 ]]
}
# Prefix first, then state: distinct state-dir arguments cannot race the same app.
# Stock lockf's fd form retains the BSD lock on our shared open description.
# Kernel release survives SIGKILL; never unlink/truncate either lock inode.
prepare_lock "$prefix/.ears-installer.lock" || exit 3
exec 8<"$prefix/.ears-installer.lock"
/usr/bin/lockf -s -t 0 8 || { printf '%s\n' 'Another installer holds the prefix lock' >&2; exit 3; }
prepare_lock "$state/installer.lock" || exit 3
exec 9<"$state/installer.lock"
/usr/bin/lockf -s -t 0 9 || { printf '%s\n' 'Another installer holds the state lock' >&2; exit 3; }
recovery_allowed=false
on_exit() {
    local status=$?
    trap - EXIT INT TERM HUP
    if $recovery_allowed && exists "$journal"; then
        if ! recover_transaction; then
            printf '%s\n' 'Recovery incomplete; journal and all remaining app copies retained. Run --recover after inspection.' >&2
            [[ "$status" != 0 ]] || status=1
        fi
    fi
    exec 9<&- 8<&-
    exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
if /usr/bin/pgrep -u "$uid" -x ears-ui >/dev/null 2>&1; then
    printf '%s\n' 'Quit eaRS before modifying or recovering this installation' >&2; exit 4
else result=$?; [[ "$result" == 1 ]] || exit 4; fi
recovery_allowed=true
recover_transaction || exit 6
[[ "$action" != recover ]] || { printf '%s\n' 'Recovery complete. No app launched or user data changed.'; exit 0; }
R_current='' R_previous='' R_previous_hash=''
if exists "$receipt"; then read_receipt "$receipt" || { printf '%s\n' 'Receipt metadata rejected' >&2; exit 5; }; fi
old_receipt=$(receipt_fingerprint) || exit 5
if [[ -n "$R_current" ]]; then verify_app "$destination" "$R_current" || exit 5
else ! exists "$destination" || { fail 'Unmanaged destination; refusing adoption/removal'; exit 5; }; fi
[[ -z "$R_previous" ]] || verify_app "$R_previous" "$R_previous_hash" || exit 5
if [[ "$action" == uninstall ]]; then
    [[ -n "$R_current" ]] || { printf '%s\n' 'Application already absent; backups and user data retained.'; exit 0; }
    pin=''
elif [[ "$action" == rollback ]]; then
    [[ -n "$R_previous" ]] || { fail 'No managed rollback backup'; exit 5; }
    source_app=$R_previous; pin=$R_previous_hash
fi
if [[ -n "$pin" ]]; then
    verify_app "$source_app" "$pin" || exit 5
    [[ "$R_current" != "$pin" ]] || { printf '%s\n' 'Already installed with trusted CDHash; nothing changed.'; exit 0; }
fi
transaction=$(/usr/bin/mktemp -d "$prefix/.ears-install.XXXXXX")
if [[ -n "$pin" ]]; then
    /usr/bin/ditto "$source_app" "$transaction/incoming.app"
    verify_app "$transaction/incoming.app" "$pin"
fi
new_previous=$R_previous new_previous_hash=$R_previous_hash
if [[ -n "$R_current" ]]; then new_previous="$transaction/old.app"; new_previous_hash=$R_current; fi
next_receipt=$(/usr/bin/mktemp "$state/.ears-receipt.XXXXXX")
receipt_json "$pin" "$new_previous" "$new_previous_hash" > "$next_receipt"
J_tx=$transaction J_old=$R_current J_new=$pin J_old_receipt=$old_receipt J_next=$next_receipt
J_next_sha=$(file_sha "$next_receipt") J_previous=$new_previous J_previous_hash=$new_previous_hash
journal_temp=$(/usr/bin/mktemp "$state/.ears-journal.XXXXXX")
journal_json > "$journal_temp"
expect_receipt "$old_receipt" || exit 5
move_absent "$journal_temp" "$journal"
# WAL published before either app rename. Recovery is selected by receipt digest,
# not an in-memory phase flag: signals immediately after any rename remain safe.
if [[ -n "$J_old" ]]; then move_absent "$destination" "$transaction/old.app"; fi
if [[ -n "$J_new" ]]; then move_absent "$transaction/incoming.app" "$destination"; fi
expect_receipt "$old_receipt" || exit 5
# Same-filesystem metadata commit, even if prefix and state are on different volumes.
/bin/mv -f -- "$next_receipt" "$receipt"
recover_transaction
printf 'Completed %s: %s\nBackups, settings, models and history retained. No application launched.\n' "$action" "$destination"
