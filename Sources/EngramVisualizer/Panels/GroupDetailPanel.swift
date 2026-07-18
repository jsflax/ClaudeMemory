import SwiftUI

/// The team-management panel: members with role menus, invite-by-email,
/// pending outbound invites (revoke + copy link), the group's project
/// registry, sub-groups, and leave/delete. A ZStack overlay in GraphView
/// (the codebase's panel idiom — no sheets), gated on
/// `groupService.selectedGroupId`.
///
/// UI gates by the viewer's own effective role for affordance only; the
/// server is always the enforcer.
struct GroupDetailPanel: View {
    @Environment(GroupService.self) private var groupService
    @Environment(AccountService.self) private var accountService
    let groupId: UUID
    let onClose: () -> Void

    @State private var inviteEmail = ""
    @State private var inviteRole: GroupService.GroupRole = .member
    @State private var newProjectName = ""
    @State private var confirmingLeave = false
    @State private var confirmingDelete = false
    @State private var copiedInviteId: UUID?

    private var group: GroupService.GroupSummary? {
        groupService.group(groupId)
    }
    private var myRole: GroupService.GroupRole {
        group.flatMap { groupService.myRole(in: $0) } ?? .member
    }
    private var isAdmin: Bool { myRole >= .admin }
    private var myUserId: UUID? { accountService.userProfile?.id }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
                .padding(.bottom, 14)

            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 20) {
                    membersBlock
                    if isAdmin {
                        inviteBlock
                        pendingInvitesBlock
                    }
                    projectsBlock
                    subGroupsBlock
                    dangerBlock
                    if let error = groupService.errorMessage {
                        Text(error)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.red.opacity(0.8))
                    }
                }
            }
        }
        .padding(20)
        .background {
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(red: 0.08, green: 0.1, blue: 0.14))
                .overlay {
                    RoundedRectangle(cornerRadius: 16)
                        .strokeBorder(.cyan.opacity(0.25), lineWidth: 1)
                }
                .shadow(color: .black.opacity(0.5), radius: 30, y: 10)
        }
        .task(id: groupId) {
            await groupService.loadGroupDetail(groupId)
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: group?.root == true ? "person.3.fill" : "person.2.fill")
                .font(.system(size: 12))
                .foregroundStyle(.cyan.opacity(0.7))
            Text(group?.name ?? "Group")
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white.opacity(0.9))
                .lineLimit(1)
            Text(myRole.rawValue)
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.cyan.opacity(0.7))
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background(Capsule().fill(.cyan.opacity(0.12)))
            Spacer()
            Button(action: onClose) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.white.opacity(0.4))
                    .font(.system(size: 16))
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Members

    private var membersBlock: some View {
        block("Members") {
            let members = groupService.membersByGroup[groupId] ?? []
            if members.isEmpty {
                Text("Loading…")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.35))
            }
            ForEach(members) { member in
                memberRow(member)
            }
        }
    }

    private func memberRow(_ member: GroupService.MemberInfo) -> some View {
        let isSelf = member.userId == myUserId
        let memberRole = GroupService.GroupRole(rawValue: member.role) ?? .member
        // Role edits: admin+ over lower roles; owner grants need owner; and
        // DIRECT members only — the server's updateMember/removeMember 404
        // on subtree-implied rows ("via sub-group" members are managed from
        // their own group's panel).
        let canEdit = isAdmin && !isSelf && member.direct && memberRole < myRole
        return HStack(spacing: 8) {
            Image(systemName: "person.circle.fill")
                .font(.system(size: 14))
                .foregroundStyle(.white.opacity(0.3))
            VStack(alignment: .leading, spacing: 1) {
                Text(member.displayName + (isSelf ? " (you)" : ""))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.8))
                    .lineLimit(1)
                if member.lastSyncAt == nil && myRole == .owner {
                    // Owner's seat-pruning signal: billed but never synced.
                    Text("never synced")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.yellow.opacity(0.5))
                }
            }
            if !member.direct {
                Text("via sub-group")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.3))
            }
            Spacer()
            if canEdit {
                Menu {
                    ForEach(GroupService.GroupRole.allCases, id: \.rawValue) { role in
                        // Granting owner requires owner.
                        if role < .owner || myRole == .owner {
                            Button(role.rawValue) {
                                Task { await groupService.setRole(groupId: groupId, userId: member.userId, role: role) }
                            }
                        }
                    }
                    Divider()
                    Button("Remove", role: .destructive) {
                        Task { await groupService.removeMember(groupId: groupId, userId: member.userId) }
                    }
                } label: {
                    Text(member.role)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.6))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
            } else {
                Text(member.role)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.4))
            }
        }
        .padding(.vertical, 3)
    }

    // MARK: - Invites

    private var inviteBlock: some View {
        block("Invite") {
            HStack(spacing: 6) {
                TextField("email@example.com", text: $inviteEmail)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 11, design: .monospaced))
                Menu {
                    Button("member") { inviteRole = .member }
                    Button("admin") { inviteRole = .admin }
                } label: {
                    Text(inviteRole.rawValue)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.6))
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                Button("Send") {
                    let email = inviteEmail
                    inviteEmail = ""
                    Task { await groupService.createInvite(groupId: groupId, email: email, role: inviteRole) }
                }
                .font(.system(size: 11, design: .monospaced))
                .disabled(!inviteEmail.contains("@"))
            }
        }
    }

    @ViewBuilder
    private var pendingInvitesBlock: some View {
        let invites = (groupService.outboundInvitesByGroup[groupId] ?? [])
        if !invites.isEmpty {
            block("Pending invites") {
                ForEach(invites) { invite in
                    HStack(spacing: 8) {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(invite.email)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(.white.opacity(0.7))
                                .lineLimit(1)
                            Text("\(invite.role) · expires \(invite.expiresAt.formatted(date: .abbreviated, time: .omitted))")
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundStyle(.white.opacity(0.35))
                        }
                        Spacer()
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(invite.url, forType: .string)
                            copiedInviteId = invite.id
                        } label: {
                            Image(systemName: copiedInviteId == invite.id ? "checkmark" : "link")
                                .font(.system(size: 10))
                                .foregroundStyle(.cyan.opacity(0.7))
                        }
                        .buttonStyle(.plain)
                        .help("Copy invite link")
                        Button {
                            Task { await groupService.revokeInvite(groupId: groupId, inviteId: invite.id) }
                        } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 10))
                                .foregroundStyle(.red.opacity(0.6))
                        }
                        .buttonStyle(.plain)
                        .help("Revoke")
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    // MARK: - Project registry (decision 13)

    private var projectsBlock: some View {
        block("Group projects") {
            let projects = groupService.projectsByGroup[groupId] ?? []
            if projects.isEmpty {
                Text("No projects registered")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.35))
            }
            ForEach(projects) { project in
                HStack(spacing: 8) {
                    Circle()
                        .fill(.cyan.opacity(0.5))
                        .frame(width: 6, height: 6)
                    Text(project.name)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.7))
                        .lineLimit(1)
                    Spacer()
                    if isAdmin {
                        Button {
                            Task { await groupService.deleteGroupProject(groupId: groupId, projectId: project.id) }
                        } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 9))
                                .foregroundStyle(.white.opacity(0.3))
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.vertical, 2)
            }
            if isAdmin {
                HStack(spacing: 6) {
                    TextField("Add project…", text: $newProjectName)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 11, design: .monospaced))
                    Button("Add") {
                        let name = newProjectName
                        newProjectName = ""
                        Task { await groupService.createGroupProject(groupId: groupId, name: name) }
                    }
                    .font(.system(size: 11, design: .monospaced))
                    .disabled(newProjectName.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
        }
    }

    // MARK: - Sub-groups

    private var subGroupsBlock: some View {
        block("Sub-groups") {
            let children = groupService.children(of: groupId)
            ForEach(children) { child in
                Button {
                    groupService.selectedGroupId = child.id
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "person.2")
                            .font(.system(size: 10))
                            .foregroundStyle(.white.opacity(0.5))
                        Text(child.name)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.7))
                            .lineLimit(1)
                        Spacer()
                        Text("\(child.memberCount)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.3))
                        Image(systemName: "chevron.right")
                            .font(.system(size: 9))
                            .foregroundStyle(.white.opacity(0.25))
                    }
                    .padding(.vertical, 3)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            if isAdmin {
                Button {
                    groupService.createGroupParentId = groupId
                    groupService.showCreateGroup = true
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "plus.circle")
                            .font(.system(size: 10))
                        Text("New sub-group")
                            .font(.system(size: 11, design: .monospaced))
                    }
                    .foregroundStyle(.cyan.opacity(0.8))
                    .padding(.vertical, 3)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
    }

    // MARK: - Leave / delete

    /// The viewer's own membership row (nil until members load).
    private var myMemberRow: GroupService.MemberInfo? {
        guard let myUserId else { return nil }
        return (groupService.membersByGroup[groupId] ?? [])
            .first { $0.userId == myUserId }
    }
    /// Sole direct owner — the server's last-owner protection would 409.
    private var isSoleOwner: Bool {
        let directOwners = (groupService.membersByGroup[groupId] ?? [])
            .filter { $0.direct && $0.role == "owner" }
        return myRole == .owner && directOwners.count <= 1
    }

    private var dangerBlock: some View {
        block("") {
            HStack(spacing: 12) {
                if let mine = myMemberRow, !mine.direct {
                    // Membership implied via a descendant — the server has
                    // no direct row here to delete; leave from the sub-group.
                    Text("membership is via a sub-group — leave from there")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.35))
                    Spacer()
                } else if isSoleOwner && !confirmingDelete {
                    Text("transfer ownership before leaving")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.35))
                    if myRole == .owner {
                        Button("Delete group") {
                            confirmingDelete = true
                        }
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.red.opacity(0.7))
                        .buttonStyle(.plain)
                    }
                    Spacer()
                } else if confirmingLeave {
                    confirmPair("Leave \(group?.name ?? "group")?") {
                        Task { await groupService.leaveGroup(groupId) }
                    } onCancel: {
                        confirmingLeave = false
                    }
                } else if confirmingDelete {
                    confirmPair("Delete \(group?.name ?? "group") and all sub-groups?") {
                        Task { await groupService.deleteGroup(groupId) }
                    } onCancel: {
                        confirmingDelete = false
                    }
                } else {
                    Button("Leave group") {
                        confirmingLeave = true
                    }
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
                    .buttonStyle(.plain)
                    if myRole == .owner {
                        Button("Delete group") {
                            confirmingDelete = true
                        }
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.red.opacity(0.7))
                        .buttonStyle(.plain)
                    }
                    Spacer()
                }
            }
        }
    }

    private func confirmPair(_ prompt: String, onConfirm: @escaping () -> Void, onCancel: @escaping () -> Void) -> some View {
        HStack(spacing: 8) {
            Text(prompt)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.white.opacity(0.6))
                .lineLimit(2)
            Spacer()
            Button("Confirm", action: onConfirm)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(.red.opacity(0.8))
                .buttonStyle(.plain)
            Button("Cancel", action: onCancel)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.white.opacity(0.5))
                .buttonStyle(.plain)
        }
    }

    // MARK: - Block helper (panel-local section)

    private func block(_ title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            if !title.isEmpty {
                Text(title.uppercased())
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.3))
                    .tracking(1.2)
            }
            content()
        }
    }
}
