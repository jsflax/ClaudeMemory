import Foundation

/// Crash-safe GPU log — fflush after every write survives hard power-off.
/// Shared between MetalForceCompute, ForceSimulation3D, and MetalGraphRenderer.
public enum GPULog {
    private nonisolated(unsafe) static var logFile: UnsafeMutablePointer<FILE>?

    /// Call once at app launch to set the log file path.
    public static func configure(path: String) {
        logFile = fopen(path, "a")
        if let f = logFile {
            fputs("\n--- APP LAUNCH \(Date()) ---\n", f)
            fflush(f)
        }
    }

    /// Write a timestamped log line. Thread-safe via fflush.
    public nonisolated static func log(_ msg: String) {
        guard let f = logFile else { return }
        let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent())
        fputs("[\(ts)] \(msg)\n", f)
        fflush(f)
    }
}
