import AuthenticationServices
import EngramModels
import GoogleSignInSwift
import Lattice
import SwiftUI

// MARK: - Account Tab

extension SidebarView {
    @ViewBuilder
    var accountContent: some View {
        if accountService.isSignedIn {
            signedInContent
                .onAppear {
                    refreshProjectCounts()
                    syncManager.refreshDaemonHealth()
                }
        } else {
            signInContent
        }
    }

    @ViewBuilder
    var signedInContent: some View {
        VStack(alignment: .leading, spacing: 24) {
            section("Account") {
                VStack(alignment: .leading, spacing: 12) {
                    HStack(spacing: 10) {
                        if let picURL = accountService.userProfile?.profilePictureUrl,
                           let url = URL(string: picURL) {
                            AsyncImage(url: url) { image in
                                image.resizable()
                                    .aspectRatio(contentMode: .fill)
                            } placeholder: {
                                Image(systemName: "person.circle.fill")
                                    .font(.system(size: 28))
                                    .foregroundStyle(.white.opacity(0.4))
                            }
                            .frame(width: 32, height: 32)
                            .clipShape(Circle())
                        } else {
                            Image(systemName: "person.circle.fill")
                                .font(.system(size: 28))
                                .foregroundStyle(.white.opacity(0.4))
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            if let name = accountService.userProfile?.fullName {
                                Text(name)
                                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.9))
                            }
                            if let sub = accountService.subscription {
                                Text(sub.displayStatus)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.6))
                                if let tier = sub.tier {
                                    Text("Engram \(tier.capitalized)")
                                        .font(.system(size: 10, design: .monospaced))
                                        .foregroundStyle(.white.opacity(0.35))
                                }
                            } else {
                                Text("Free")
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundStyle(.white.opacity(0.6))
                            }
                        }
                    }

                    Button("Sign Out") {
                        syncManager.disconnectSync()
                        accountService.signOut()
                    }
                    .font(.system(size: 11, design: .monospaced))

                    if syncManager.isSyncing {
                        VStack(alignment: .leading, spacing: 6) {
                            if let ipc = syncManager.ipcProgress {
                                syncProgressRow(label: "IPC", progress: ipc, color: .cyan)
                            }
                            // Connection health outranks progress: a dead WSS
                            // with nothing pending used to render as "Synced".
                            if let health = syncManager.daemonHealth, !health.isHealthy, !health.isStale {
                                daemonProblemRow(health: health)
                            } else if let wss = syncManager.wssProgress {
                                syncProgressRow(label: "WSS", progress: wss, color: .orange)
                            }
                        }
                    }
                }
            }

            if syncManager.isSyncing {
                section("Projects") {
                    VStack(alignment: .leading, spacing: 4) {
                        if projects.isEmpty {
                            Text("No projects yet")
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(.white.opacity(0.35))
                        } else {
                            ForEach(projects, id: \.self) { project in
                                syncProjectRow(project)
                            }
                        }

                        if let status = syncManager.statusMessage {
                            Text(status)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(.cyan.opacity(0.7))
                                .padding(.top, 4)
                        }
                    }
                }
            }

            #if DEBUG
            section("Endpoint") {
                TextField("https://engram.io", text: $accountService.endpoint)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 11, design: .monospaced))
            }
            #endif
        }
    }


    func daemonProblemRow(health: SyncManager.DaemonHealth) -> some View {
        HStack(spacing: 6) {
            Text("WSS")
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundStyle(.white.opacity(0.35))
                .frame(width: 24, alignment: .leading)
            Circle()
                .fill(health.state == "waiting_for_auth" ? Color.yellow : Color.red)
                .frame(width: 6, height: 6)
            Text(health.state == "waiting_for_auth"
                    ? "Sign in required"
                    : (health.detail ?? health.state))
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.white.opacity(0.5))
                .lineLimit(1)
            Spacer()
        }
    }

    func syncProgressRow(label: String, progress: Lattice.SyncProgress, color: Color) -> some View {
        let isUploading = progress.isUploading
        return HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundStyle(.white.opacity(0.35))
                .frame(width: 24, alignment: .leading)

            if isUploading {
                ProgressView()
                    .controlSize(.mini)
                    .scaleEffect(0.5)
                Text("\(progress.acked)/\(progress.totalUpload)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.4))
                Spacer()
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(.white.opacity(0.1))
                        RoundedRectangle(cornerRadius: 2)
                            .fill(color.opacity(0.6))
                            .frame(width: geo.size.width * progress.uploadFraction)
                            .animation(.easeInOut(duration: 0.2), value: progress.uploadFraction)
                    }
                }
                .frame(width: 50, height: 3)
            } else {
                Circle()
                    .fill(.green)
                    .frame(width: 5, height: 5)
                Text("Synced")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.35))
                Spacer()
            }
        }
    }

    func syncProjectRow(_ project: String) -> some View {
        let isOn = syncConfigs.first(where: { $0.project == project })?.policy == .sync
        let count = projectCounts[project] ?? 0
        return Button {
            syncManager.toggleProject(project)
        } label: {
            HStack(spacing: 8) {
                togglePill(isOn: isOn)

                Text(project)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.7))
                    .lineLimit(1)

                Spacer()

                Text("\(count)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.3))
            }
            .padding(.vertical, 5)
            .padding(.horizontal, 8)
            .background(
                RoundedRectangle(cornerRadius: 5)
                    .fill(.white.opacity(0.03))
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    var signInContent: some View {
        VStack(alignment: .leading, spacing: 24) {
            section("Sign In") {
                VStack(spacing: 12) {
                    Text("Sign in to enable cloud sync and backup.")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.5))
                        .fixedSize(horizontal: false, vertical: true)

                    SignInWithAppleButton(.signIn) { request in
                        request.requestedScopes = [.email, .fullName]
                    } onCompletion: { result in
                        Task { await accountService.handleAppleSignIn(result: result) }
                    }
                    .frame(height: 36)

                    GoogleSignInButton {
                        Task { await accountService.signInWithGoogle() }
                    }
                    .frame(height: 36)

                    HStack(spacing: 8) {
                        Rectangle().frame(height: 1).foregroundStyle(.white.opacity(0.1))
                        Text("or")
                            .font(.system(size: 10))
                            .foregroundStyle(.white.opacity(0.3))
                        Rectangle().frame(height: 1).foregroundStyle(.white.opacity(0.1))
                    }

                    VStack(spacing: 8) {
                        TextField("Email", text: $email)
                            .textFieldStyle(.roundedBorder)
                            .textContentType(.emailAddress)
                            .font(.system(size: 12))

                        SecureField("Password", text: $password)
                            .textFieldStyle(.roundedBorder)
                            .textContentType(isRegistering ? .newPassword : .password)
                            .font(.system(size: 12))
                    }

                    if let error = accountService.errorMessage {
                        Text(error)
                            .font(.system(size: 10))
                            .foregroundStyle(.red.opacity(0.8))
                    }

                    Button(isRegistering ? "Create Account" : "Sign In") {
                        Task {
                            if isRegistering {
                                await accountService.register(email: email, password: password)
                            } else {
                                await accountService.signIn(email: email, password: password)
                            }
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .font(.system(size: 12, design: .monospaced))
                    .disabled(email.isEmpty || password.isEmpty || accountService.isLoading)

                    Button(isRegistering ? "Have an account? Sign in" : "No account? Register") {
                        isRegistering.toggle()
                        accountService.errorMessage = nil
                    }
                    .font(.system(size: 10, design: .monospaced))
                    .buttonStyle(.plain)
                    .foregroundStyle(.cyan.opacity(0.8))
                }
            }

            #if DEBUG
            section("Server") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Endpoint")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.35))
                    TextField("https://engram.io", text: $accountService.endpoint)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 11, design: .monospaced))
                }
            }
            #endif
        }
    }
}
