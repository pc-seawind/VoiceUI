# ruff: noqa: E501
from __future__ import annotations

import base64
import json
import subprocess
import sys
from typing import Any

from voiceui.audio import resolve_sounddevice_device


def resolve_output_device_name(device: str | int | None) -> str:
    if device is None or device == "default":
        return ""
    if isinstance(device, str):
        stripped = device.strip()
        if not stripped:
            return ""
        if stripped.isdigit():
            device = int(stripped)
        else:
            try:
                import sounddevice as sd  # type: ignore[import-untyped]

                resolved_device = resolve_sounddevice_device(sd, stripped, kind="output")
                info = sd.query_devices(resolved_device, "output")
                name = str(info.get("name") or "").strip()
                if name:
                    return name
            except ImportError:
                return stripped
            except RuntimeError as exc:
                if "ambiguous" in str(exc).lower():
                    raise
                return stripped
            return stripped

    try:
        import sounddevice as sd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "System volume control for a numbered audio device requires sounddevice."
        ) from exc

    info = sd.query_devices(device, "output")
    name = str(info.get("name") or "").strip()
    if not name:
        raise RuntimeError(f"Could not resolve output device name for index: {device}")
    return name


def set_system_output_volume(
    *,
    device: str | int | None = None,
    volume_percent: float | None = None,
    relative_percent: float | None = None,
    muted: bool | None = None,
) -> dict[str, Any]:
    if volume_percent is None and relative_percent is None and muted is None:
        raise RuntimeError("volume_percent, relative_percent, or muted is required.")
    device_name = resolve_output_device_name(device)
    return _run_windows_volume_script(
        device_name=device_name,
        has_device_name=bool(device_name),
        has_absolute=volume_percent is not None,
        absolute_scalar=_percent_to_scalar(volume_percent or 0),
        has_relative=relative_percent is not None,
        relative_scalar=_percent_to_scalar(relative_percent or 0),
        has_mute=muted is not None,
        muted=bool(muted),
    )


def get_system_output_volume(*, device: str | int | None = None) -> dict[str, Any]:
    device_name = resolve_output_device_name(device)
    return _run_windows_volume_script(
        device_name=device_name,
        has_device_name=bool(device_name),
        has_absolute=False,
        absolute_scalar=0,
        has_relative=False,
        relative_scalar=0,
        has_mute=False,
        muted=False,
    )


def _percent_to_scalar(value: float) -> float:
    numeric = float(value)
    if -1.0 < numeric < 1.0 and numeric != 0.0:
        return numeric
    return numeric / 100.0


def _run_windows_volume_script(
    *,
    device_name: str,
    has_device_name: bool,
    has_absolute: bool,
    absolute_scalar: float,
    has_relative: bool,
    relative_scalar: float,
    has_mute: bool,
    muted: bool,
) -> dict[str, Any]:
    if sys.platform != "win32":
        raise RuntimeError("System output volume control is currently only supported on Windows.")

    script = _volume_script(
        device_name=device_name,
        has_device_name=has_device_name,
        has_absolute=has_absolute,
        absolute_scalar=absolute_scalar,
        has_relative=has_relative,
        relative_scalar=relative_scalar,
        has_mute=has_mute,
        muted=muted,
    )
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(error or f"PowerShell exited with {completed.returncode}")
    output = completed.stdout.strip().splitlines()
    if not output:
        raise RuntimeError("PowerShell did not return volume data.")
    data = json.loads(output[-1])
    if not isinstance(data, dict):
        raise RuntimeError("PowerShell returned invalid volume data.")
    return data


def _volume_script(
    *,
    device_name: str,
    has_device_name: bool,
    has_absolute: bool,
    absolute_scalar: float,
    has_relative: bool,
    relative_scalar: float,
    has_mute: bool,
    muted: bool,
) -> str:
    target = _powershell_here_string(device_name)
    return f"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$targetName = {target}
$hasDeviceName = ${str(has_device_name).lower()}
$hasAbsolute = ${str(has_absolute).lower()}
$absoluteScalar = [single]{absolute_scalar}
$hasRelative = ${str(has_relative).lower()}
$relativeScalar = [single]{relative_scalar}
$hasMute = ${str(has_mute).lower()}
$muted = ${str(muted).lower()}

$code = @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace VoiceUISystemVolume {{
    public enum EDataFlow {{ eRender = 0, eCapture = 1, eAll = 2 }}
    public enum ERole {{ eConsole = 0, eMultimedia = 1, eCommunications = 2 }}
    [Flags]
    public enum DeviceState {{ Active = 0x1, Disabled = 0x2, NotPresent = 0x4, Unplugged = 0x8, All = 0xF }}

    [ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IMMDeviceEnumerator {{
        int EnumAudioEndpoints(EDataFlow dataFlow, DeviceState stateMask, out IMMDeviceCollection devices);
        int GetDefaultAudioEndpoint(EDataFlow dataFlow, ERole role, out IMMDevice endpoint);
        int GetDevice([MarshalAs(UnmanagedType.LPWStr)] string id, out IMMDevice device);
        int RegisterEndpointNotificationCallback(IntPtr client);
        int UnregisterEndpointNotificationCallback(IntPtr client);
    }}

    [ComImport, Guid("0BD7A1BE-7A1A-44DB-8397-CC5392387B5E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IMMDeviceCollection {{
        int GetCount(out uint count);
        int Item(uint index, out IMMDevice device);
    }}

    [ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IMMDevice {{
        int Activate(ref Guid iid, int clsCtx, IntPtr activationParams, [MarshalAs(UnmanagedType.IUnknown)] out object endpointVolume);
        int OpenPropertyStore(int access, out IPropertyStore properties);
        int GetId([MarshalAs(UnmanagedType.LPWStr)] out string id);
        int GetState(out DeviceState state);
    }}

    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IPropertyStore {{
        int GetCount(out uint count);
        int GetAt(uint propertyIndex, out PropertyKey key);
        int GetValue(ref PropertyKey key, out PropVariant value);
        int SetValue(ref PropertyKey key, ref PropVariant value);
        int Commit();
    }}

    [StructLayout(LayoutKind.Sequential)]
    public struct PropertyKey {{ public Guid fmtid; public uint pid; }}

    [StructLayout(LayoutKind.Sequential)]
    public struct PropVariant {{
        public ushort vt;
        public ushort wReserved1;
        public ushort wReserved2;
        public ushort wReserved3;
        public IntPtr p;
        public int p2;
        public string GetString() {{ return vt == 31 ? Marshal.PtrToStringUni(p) : String.Empty; }}
    }}

    [ComImport, Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IAudioEndpointVolume {{
        int RegisterControlChangeNotify(IntPtr client);
        int UnregisterControlChangeNotify(IntPtr client);
        int GetChannelCount(out uint channelCount);
        int SetMasterVolumeLevel(float levelDb, ref Guid eventContext);
        int SetMasterVolumeLevelScalar(float level, ref Guid eventContext);
        int GetMasterVolumeLevel(out float levelDb);
        int GetMasterVolumeLevelScalar(out float level);
        int SetChannelVolumeLevel(uint channel, float levelDb, ref Guid eventContext);
        int SetChannelVolumeLevelScalar(uint channel, float level, ref Guid eventContext);
        int GetChannelVolumeLevel(uint channel, out float levelDb);
        int GetChannelVolumeLevelScalar(uint channel, out float level);
        int SetMute([MarshalAs(UnmanagedType.Bool)] bool isMuted, ref Guid eventContext);
        int GetMute(out bool isMuted);
        int GetVolumeStepInfo(out uint step, out uint stepCount);
        int VolumeStepUp(ref Guid eventContext);
        int VolumeStepDown(ref Guid eventContext);
        int QueryHardwareSupport(out uint hardwareSupportMask);
        int GetVolumeRange(out float minDb, out float maxDb, out float incrementDb);
    }}

    public class VolumeResult {{
        public string device = "";
        public double before_percent;
        public double after_percent;
        public bool before_muted;
        public bool after_muted;
    }}

    public static class EndpointVolume {{
        static readonly Guid EnumeratorClsid = new Guid("BCDE0395-E52F-467C-8E3D-C4579291692E");
        static readonly Guid EndpointVolumeIid = new Guid("5CDF2C82-841E-4546-9722-0CF74078229A");
        static readonly PropertyKey FriendlyNameKey = new PropertyKey {{
            fmtid = new Guid("A45C254E-DF1C-4EFD-8020-67D146A850E0"),
            pid = 14
        }};

        public static VolumeResult Apply(string exactName, bool hasDeviceName, float absoluteScalar, bool hasAbsolute, float relativeScalar, bool hasRelative, bool muted, bool hasMute) {{
            var item = hasDeviceName ? FindByExactName(exactName) : DefaultRenderEndpoint();
            object obj;
            Guid iid = EndpointVolumeIid;
            item.Device.Activate(ref iid, 23, IntPtr.Zero, out obj);
            var volume = (IAudioEndpointVolume)obj;
            float before;
            bool beforeMuted;
            volume.GetMasterVolumeLevelScalar(out before);
            volume.GetMute(out beforeMuted);

            float target = before;
            if (hasAbsolute) target = absoluteScalar;
            if (hasRelative) target += relativeScalar;
            target = Math.Max(0, Math.Min(1, target));

            Guid ctx = Guid.Empty;
            if (hasAbsolute || hasRelative) volume.SetMasterVolumeLevelScalar(target, ref ctx);
            if (hasMute) volume.SetMute(muted, ref ctx);

            float after;
            bool afterMuted;
            volume.GetMasterVolumeLevelScalar(out after);
            volume.GetMute(out afterMuted);
            return new VolumeResult {{
                device = item.Name,
                before_percent = Math.Round(before * 100, 1),
                after_percent = Math.Round(after * 100, 1),
                before_muted = beforeMuted,
                after_muted = afterMuted
            }};
        }}

        static EndpointItem DefaultRenderEndpoint() {{
            var enumerator = CreateEnumerator();
            IMMDevice device;
            enumerator.GetDefaultAudioEndpoint(EDataFlow.eRender, ERole.eMultimedia, out device);
            return new EndpointItem {{ Name = FriendlyName(device), Device = device }};
        }}

        static EndpointItem FindByExactName(string exactName) {{
            foreach (var item in EnumerateRenderEndpoints()) {{
                if (String.Equals(item.Name, exactName, StringComparison.Ordinal)) return item;
            }}
            throw new InvalidOperationException("Render endpoint not found: " + exactName);
        }}

        static IEnumerable<EndpointItem> EnumerateRenderEndpoints() {{
            var enumerator = CreateEnumerator();
            IMMDeviceCollection collection;
            enumerator.EnumAudioEndpoints(EDataFlow.eRender, DeviceState.Active, out collection);
            uint count;
            collection.GetCount(out count);
            for (uint i = 0; i < count; i++) {{
                IMMDevice device;
                collection.Item(i, out device);
                yield return new EndpointItem {{ Name = FriendlyName(device), Device = device }};
            }}
        }}

        static IMMDeviceEnumerator CreateEnumerator() {{
            var type = Type.GetTypeFromCLSID(EnumeratorClsid);
            return (IMMDeviceEnumerator)Activator.CreateInstance(type);
        }}

        static string FriendlyName(IMMDevice device) {{
            IPropertyStore props;
            device.OpenPropertyStore(0, out props);
            PropVariant value;
            var key = FriendlyNameKey;
            props.GetValue(ref key, out value);
            return value.GetString();
        }}

        class EndpointItem {{
            public string Name;
            public IMMDevice Device;
        }}
    }}
}}
'@

Add-Type -TypeDefinition $code
$result = [VoiceUISystemVolume.EndpointVolume]::Apply($targetName, $hasDeviceName, $absoluteScalar, $hasAbsolute, $relativeScalar, $hasRelative, $muted, $hasMute)
$result | ConvertTo-Json -Compress
""".strip()


def _powershell_here_string(value: str) -> str:
    if not value:
        return "''"
    if "\n" in value or "\r" in value or "'@" in value:
        raise RuntimeError("Unsupported PowerShell string value.")
    return f"@'\n{value}\n'@"
