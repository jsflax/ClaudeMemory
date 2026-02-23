import SwiftUI

/// Toolbar indicator showing cloud sync status.
struct SyncStatusView: View {
    let accountService: AccountService

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)

            Text(statusLabel)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .help(statusHelp)
    }

    private var statusColor: Color {
        guard accountService.isSignedIn else { return .gray }
        guard let sub = accountService.subscription else { return .gray }
        switch sub.status {
        case "active", "trialing": return .green
        case "past_due": return .yellow
        default: return .red
        }
    }

    private var statusLabel: String {
        guard accountService.isSignedIn else { return "Local" }
        guard let sub = accountService.subscription else { return "Free" }
        if sub.isActive { return "Synced" }
        return sub.displayStatus
    }

    private var statusHelp: String {
        guard accountService.isSignedIn else {
            return "Not signed in. Memories are stored locally only."
        }
        guard let sub = accountService.subscription else {
            return "Free tier. Upgrade to Pro for cloud sync."
        }
        if sub.isActive {
            return "Cloud sync is active. Memories are backed up."
        }
        return "Subscription status: \(sub.displayStatus)"
    }
}
