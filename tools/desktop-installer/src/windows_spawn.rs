//! Detached Windows creation with only owned NUL stdio handles inherited.
//! `Stdio::null()` alone does not exclude the launcher's other inheritable handles.
use std::ffi::{OsStr, OsString};
use std::io;
use std::marker::PhantomData;
use std::mem::{size_of, size_of_val};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::Path;
use std::ptr::{null, null_mut};
use windows_sys::Win32::Foundation::{
    ERROR_INSUFFICIENT_BUFFER, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
};
use windows_sys::Win32::System::Threading::{
    CreateProcessW, DeleteProcThreadAttributeList, InitializeProcThreadAttributeList,
    UpdateProcThreadAttribute, CREATE_NEW_PROCESS_GROUP, CREATE_UNICODE_ENVIRONMENT,
    DETACHED_PROCESS, EXTENDED_STARTUPINFO_PRESENT, LPPROC_THREAD_ATTRIBUTE_LIST,
    PROCESS_INFORMATION, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, STARTF_USESTDHANDLES, STARTUPINFOEXW,
};

fn wide_z(value: &OsStr) -> io::Result<Vec<u16>> {
    let mut wide: Vec<_> = value.encode_wide().collect();
    if wide.contains(&0) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "embedded NUL"));
    }
    wide.push(0);
    Ok(wide)
}

// argv[0] has different parsing rules from later arguments. An executable path
// cannot contain quotes; later arguments use the CRT backslash/quote rules.
fn command_line(path: &Path, args: &[OsString]) -> io::Result<Vec<u16>> {
    let program = wide_z(path.as_os_str())?;
    if program.contains(&(b'"' as u16)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "quoted executable path",
        ));
    }
    let mut line = vec![b'"' as u16];
    line.extend_from_slice(&program[..program.len() - 1]);
    line.push(b'"' as u16);
    for arg in args {
        line.extend([b' ' as u16, b'"' as u16]);
        let mut slashes = 0;
        for unit in arg.encode_wide() {
            match unit {
                0 => return Err(io::Error::new(io::ErrorKind::InvalidInput, "embedded NUL")),
                92 => slashes += 1,
                _ => {
                    let count = if unit == 34 { slashes * 2 + 1 } else { slashes };
                    line.extend(std::iter::repeat_n(92, count));
                    line.push(unit);
                    slashes = 0;
                }
            }
        }
        line.extend(std::iter::repeat_n(92, slashes * 2));
        line.push(b'"' as u16);
    }
    if line.len() >= 32767 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Windows command line too long",
        ));
    }
    line.push(0);
    Ok(line)
}

fn null_stdio() -> io::Result<OwnedHandle> {
    let name = wide_z(OsStr::new(r"\\.\NUL"))?;
    let attributes = SECURITY_ATTRIBUTES {
        nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: null_mut(),
        bInheritHandle: 1,
    };
    // SAFETY: terminated name and security attributes live through the call.
    let handle = unsafe {
        CreateFileW(
            name.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            &attributes,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: successful CreateFileW returns a new owned kernel handle.
    Ok(unsafe { OwnedHandle::from_raw_handle(handle) })
}

struct AttributeList<'a> {
    storage: Vec<usize>,
    _handles: PhantomData<&'a [HANDLE]>,
}

impl<'a> AttributeList<'a> {
    fn new(handles: &'a [HANDLE]) -> io::Result<Self> {
        let mut bytes = 0;
        // SAFETY: the documented first call queries the required buffer size.
        unsafe { InitializeProcThreadAttributeList(null_mut(), 1, 0, &mut bytes) };
        let error = io::Error::last_os_error();
        if bytes == 0 || error.raw_os_error() != Some(ERROR_INSUFFICIENT_BUFFER as i32) {
            return Err(error);
        }
        let mut storage = vec![0usize; bytes.div_ceil(size_of::<usize>())];
        let ptr = storage.as_mut_ptr().cast();
        // SAFETY: pointer-aligned allocation covers the queried size.
        if unsafe { InitializeProcThreadAttributeList(ptr, 1, 0, &mut bytes) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut list = Self {
            storage,
            _handles: PhantomData,
        };
        // SAFETY: the list is initialized; the borrowed handle array remains live
        // until after CreateProcessW. No pseudo or unrelated handle is included.
        if unsafe {
            UpdateProcThreadAttribute(
                list.as_mut_ptr(),
                0,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
                handles.as_ptr().cast(),
                size_of_val(handles),
                null_mut(),
                null(),
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(list)
    }

    fn as_mut_ptr(&mut self) -> LPPROC_THREAD_ATTRIBUTE_LIST {
        self.storage.as_mut_ptr().cast()
    }
}

impl Drop for AttributeList<'_> {
    fn drop(&mut self) {
        // SAFETY: only successfully initialized lists construct this guard; its
        // backing storage is still live and is freed after this destructor.
        unsafe { DeleteProcThreadAttributeList(self.as_mut_ptr()) };
    }
}

pub(super) fn spawn(path: &Path, cwd: &Path, args: &[OsString]) -> io::Result<u32> {
    if !path.is_absolute() || !cwd.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "absolute executable and cwd required",
        ));
    }
    let program = wide_z(path.as_os_str())?;
    let directory = wide_z(cwd.as_os_str())?;
    let mut line = command_line(path, args)?;
    let nul = null_stdio()?;
    let handles = [nul.as_raw_handle()];
    let mut attributes = AttributeList::new(&handles)?;
    let mut startup = STARTUPINFOEXW::default();
    startup.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = handles[0];
    startup.StartupInfo.hStdOutput = handles[0];
    startup.StartupInfo.hStdError = handles[0];
    startup.lpAttributeList = attributes.as_mut_ptr();
    let mut info = PROCESS_INFORMATION::default();
    // SAFETY: all buffers, NUL handles and the allowlist live through creation.
    // TRUE is required with HANDLE_LIST, which restricts inheritance to NUL.
    // A null environment inherits the existing environment, not a synthesized one.
    if unsafe {
        CreateProcessW(
            program.as_ptr(),
            line.as_mut_ptr(),
            null(),
            null(),
            1,
            DETACHED_PROCESS
                | CREATE_NEW_PROCESS_GROUP
                | CREATE_UNICODE_ENVIRONMENT
                | EXTENDED_STARTUPINFO_PRESENT,
            null(),
            directory.as_ptr(),
            &startup.StartupInfo,
            &mut info,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: successful creation returns two owned handles. Closing them neither
    // waits for nor terminates the child; all failure paths above also use RAII.
    let _process = unsafe { OwnedHandle::from_raw_handle(info.hProcess) };
    let _thread = unsafe { OwnedHandle::from_raw_handle(info.hThread) };
    Ok(info.dwProcessId)
}
