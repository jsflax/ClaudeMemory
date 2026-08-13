import Foundation

/// Minimal reader for SQLite's WAL-index (the `-shm` file) — enough to
/// answer ONE question cheaply, with no SQLite API, no connection, and no
/// locks: how far behind the checkpointer is (`nBackfill`) relative to the
/// WAL head (`mxFrame`). A WAL that keeps growing while nBackfill sits far
/// below mxFrame is PINNED — some reader holds a read mark that checkpoints
/// cannot pass (Aug 2026 incident: orphaned read keepers held the hub WAL
/// at 16GB, twice).
///
/// Layout (sqlite.org/walformat.html, "WAL-Index Format"): the file opens
/// with TWO identical 48-byte copies of `WalIndexHdr` (offsets 0 and 48) —
/// writers update copy 2 first, then copy 1, so a lock-free reader trusts
/// the header only when BOTH copies agree. `WalCkptInfo` follows at offset
/// 96, and its first 32-bit field is `nBackfill`. All fields are in the
/// HOST's native byte order: the wal-index is a memory-mapped struct, not
/// a portable file (unlike the big-endian `-wal` header).
public enum WalIndex {

    public struct Snapshot: Sendable, Equatable {
        /// Index of the last valid frame in the WAL.
        public let mxFrame: UInt32
        /// Number of WAL frames already copied back into the database.
        public let nBackfill: UInt32

        public init(mxFrame: UInt32, nBackfill: UInt32) {
            self.mxFrame = mxFrame
            self.nBackfill = nBackfill
        }
    }

    /// Byte offsets into the -shm file (see layout note above).
    private static let headerCopySize = 48
    private static let isInitOffset = 12
    private static let mxFrameOffset = 16
    private static let checkpointInfoOffset = 96
    /// Through nBackfill — the only WalCkptInfo field this reader needs.
    private static let minimumLength = 100

    /// Read a consistent (mxFrame, nBackfill) snapshot from `shmPath`.
    ///
    /// Returns nil when the file is missing or short, the header is not
    /// initialized, or the two header copies disagree (a writer is
    /// mid-update — the caller simply tries again next pass).
    public static func read(shmPath: String) -> Snapshot? {
        guard let data = FileManager.default.contents(atPath: shmPath),
              data.count >= minimumLength else { return nil }
        guard data[isInitOffset] != 0,
              data[headerCopySize + isInitOffset] != 0 else { return nil }
        let mxFrame1 = loadUInt32(data, at: mxFrameOffset)
        let mxFrame2 = loadUInt32(data, at: headerCopySize + mxFrameOffset)
        guard mxFrame1 == mxFrame2 else { return nil }
        return Snapshot(mxFrame: mxFrame1,
                        nBackfill: loadUInt32(data, at: checkpointInfoOffset))
    }

    private static func loadUInt32(_ data: Data, at offset: Int) -> UInt32 {
        data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
    }
}
