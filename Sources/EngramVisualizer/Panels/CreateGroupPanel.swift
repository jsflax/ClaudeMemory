import SwiftUI

/// Centered overlay for creating a group (or sub-group). Parent picker
/// offers groups where the viewer's effective role is admin+ (child
/// creation needs admin on the parent, server-enforced); pre-seeded from
/// `groupService.createGroupParentId` when opened from a detail panel.
struct CreateGroupPanel: View {
    @Environment(GroupService.self) private var groupService
    let onClose: () -> Void

    @State private var name = ""
    @State private var parentId: UUID?
    @State private var isCreating = false

    private var parentCandidates: [GroupService.GroupSummary] {
        groupService.groups.filter { groupService.canAdmin($0) }.sorted { $0.name < $1.name }
    }
    private var parentName: String {
        parentId.flatMap { groupService.group($0)?.name } ?? "None (top-level)"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text(parentId == nil ? "New Group" : "New Sub-group")
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.9))
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.white.opacity(0.4))
                        .font(.system(size: 16))
                }
                .buttonStyle(.plain)
            }

            TextField("Group name", text: $name)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12, design: .monospaced))

            if !parentCandidates.isEmpty {
                HStack(spacing: 8) {
                    Text("Parent")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.5))
                    Menu {
                        Button("None (top-level)") { parentId = nil }
                        ForEach(parentCandidates) { candidate in
                            Button(candidate.name) { parentId = candidate.id }
                        }
                    } label: {
                        Text(parentName)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.8))
                    }
                    .menuStyle(.borderlessButton)
                }
            }

            if let error = groupService.errorMessage {
                Text(error)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.red.opacity(0.8))
                    .lineLimit(2)
            }

            HStack {
                Spacer()
                Button("Create") {
                    guard !isCreating else { return }
                    isCreating = true
                    Task {
                        let created = await groupService.createGroup(name: name, parentId: parentId)
                        isCreating = false
                        if let created {
                            groupService.selectedGroupId = created.id
                            onClose()
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .font(.system(size: 12, design: .monospaced))
                .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isCreating)
            }
        }
        .padding(20)
        .frame(width: 340)
        .background {
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(red: 0.08, green: 0.1, blue: 0.14))
                .overlay {
                    RoundedRectangle(cornerRadius: 16)
                        .strokeBorder(.cyan.opacity(0.25), lineWidth: 1)
                }
                .shadow(color: .black.opacity(0.5), radius: 30, y: 10)
        }
        .onAppear {
            parentId = groupService.createGroupParentId
        }
    }
}
