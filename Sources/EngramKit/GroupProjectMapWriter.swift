import Foundation
import Lattice
import EngramModels

/// Upserts a member's local→group project mapping into a group spoke
/// (decision 13's default: group project named after the local project).
///
/// This is the ONE shared write path for `GroupProjectMap` rows — used by
/// the visualizer's exposure toggle, the `memory-sync expose` CLI, and the
/// daemon's spoke-open backstop. Before it was shared, only the visualizer
/// wrote the map, so headless members recalled with identity-only project
/// resolution: no cross-member boost, no canonical clustering.
public enum GroupProjectMapWriter {

    /// Upsert `localProject → groupProject(defaulting to localProject)` for
    /// `me` into the spoke at `spokePath`. Returns false when the spoke file
    /// does not exist yet (daemon hasn't created it) — callers treat that as
    /// "the daemon backstop will write it at next spoke open", not an error.
    @discardableResult
    public static func upsert(spokePath: String,
                              me: UUID,
                              localProject: String,
                              log: (String) -> Void = { print($0) }) -> Bool {
        guard FileManager.default.fileExists(atPath: spokePath) else { return false }
        // migration: must match every other Engram DB open — without it the
        // open runs at target_schema_version 1 and LatticeCore rejects a
        // daemon-created (current-version) spoke as "newer than this binary
        // supports".
        guard let spoke = try? Lattice(
            Memory.self, Edge.self, GroupProjectMap.self,
            configuration: .init(fileURL: URL(fileURLWithPath: spokePath),
                                 migration: engramMigrations)
        ) else {
            log("[GroupProjectMapWriter] spoke open failed for \(spokePath) — upsert skipped")
            return false
        }
        return upsert(into: spoke, me: me, localProject: localProject, log: log)
    }

    /// Same upsert against an already-open spoke lattice (the daemon holds
    /// its spokes open; re-opening the file under a live IPC relay handle is
    /// unnecessary churn).
    @discardableResult
    public static func upsert(into spoke: Lattice,
                              me: UUID,
                              localProject: String,
                              log: (String) -> Void = { print($0) }) -> Bool {
        if let existing = spoke.objects(GroupProjectMap.self)
            .where({ $0.memberUserId == me && $0.localProject == localProject })
            .first {
            if existing.groupProject != localProject {
                existing.groupProject = localProject
                existing.updatedAt = Date()
            }
            return true
        }
        do {
            try spoke.add(GroupProjectMap(memberUserId: me,
                                          localProject: localProject,
                                          groupProject: localProject))
            return true
        } catch {
            log("[GroupProjectMapWriter] insert failed: \(error)")
            return false
        }
    }
}
