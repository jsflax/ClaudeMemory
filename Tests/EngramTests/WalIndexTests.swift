import EngramKit
import EngramModels
import Foundation
import Lattice
import Testing

// ============================================================================
// H4 (Aug 2026 incident): the daemon's hourly WAL self-diagnosis reads
// (mxFrame, nBackfill) straight from the -shm wal-index header — two 48-byte
// header copies that must agree, then WalCkptInfo.nBackfill at offset 96,
// all in host byte order. These tests pin that layout against synthetic
// files and against a real Lattice-written -shm.
// ============================================================================

@Suite("WAL-index reader (H4)")
struct WalIndexTests {

    /// Build a synthetic -shm: isInit bytes at 12 and 60, mxFrame copies at
    /// 16 and 64, nBackfill at 96 — everything else zero.
    private func writeShm(mxFrame1: UInt32, mxFrame2: UInt32, nBackfill: UInt32,
                          isInit: UInt8 = 1, length: Int = 32768) throws -> String {
        var data = Data(count: length)
        if length > 60 {
            data[12] = isInit
            data[48 + 12] = isInit
        }
        func put(_ value: UInt32, at offset: Int) {
            guard length >= offset + 4 else { return }
            withUnsafeBytes(of: value) { data.replaceSubrange(offset..<offset + 4, with: $0) }
        }
        put(mxFrame1, at: 16)
        put(mxFrame2, at: 48 + 16)
        put(nBackfill, at: 96)
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("walindex-test-\(UUID().uuidString)-shm").path
        try data.write(to: URL(fileURLWithPath: path))
        return path
    }

    @Test func agreeingHeaderCopies_parse() throws {
        let path = try writeShm(mxFrame1: 123_456, mxFrame2: 123_456, nBackfill: 42)
        let snapshot = try #require(WalIndex.read(shmPath: path))
        #expect(snapshot.mxFrame == 123_456)
        #expect(snapshot.nBackfill == 42)
    }

    /// Writers update copy 2 first, then copy 1 — disagreeing copies mean a
    /// torn read, and a diagnosis built on one would be garbage.
    @Test func disagreeingHeaderCopies_areRejected() throws {
        let path = try writeShm(mxFrame1: 200, mxFrame2: 100, nBackfill: 0)
        #expect(WalIndex.read(shmPath: path) == nil)
    }

    @Test func uninitializedHeader_isRejected() throws {
        let path = try writeShm(mxFrame1: 10, mxFrame2: 10, nBackfill: 0, isInit: 0)
        #expect(WalIndex.read(shmPath: path) == nil)
    }

    @Test func shortOrMissingFile_isRejected() throws {
        let short = try writeShm(mxFrame1: 1, mxFrame2: 1, nBackfill: 0, length: 64)
        #expect(WalIndex.read(shmPath: short) == nil)
        #expect(WalIndex.read(shmPath: "/nonexistent/path-shm") == nil)
    }

    /// Against a REAL wal-index: rows written through Lattice land as WAL
    /// frames, so with the connection still open mxFrame must be positive,
    /// nBackfill can never exceed it, and the header copies must agree.
    @Test func realLatticeShm_readsConsistently() throws {
        let dbURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("walindex-real-\(UUID().uuidString).sqlite")
        let lattice = try Lattice(
            Memory.self, Edge.self, Checkpoint.self, HookState.self, SyncConfig.self,
            configuration: .init(fileURL: dbURL))
        for i in 0..<25 {
            try lattice.add(Memory(content: "wal frame probe \(i)", project: "walindex"))
        }
        let snapshot = try #require(WalIndex.read(shmPath: dbURL.path + "-shm"),
                                    "unreadable -shm for a live Lattice database")
        #expect(snapshot.mxFrame > 0)
        #expect(snapshot.nBackfill <= snapshot.mxFrame)
        withExtendedLifetime(lattice) {}
    }
}
