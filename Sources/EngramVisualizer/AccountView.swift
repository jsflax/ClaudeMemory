import AuthenticationServices
import GoogleSignInSwift
import SwiftUI

/// Account management view: sign in/up, subscription status, sync settings.
struct AccountView: View {
    @Bindable var accountService: AccountService

    @State private var email = ""
    @State private var password = ""
    @State private var isRegistering = false

    var body: some View {
        VStack(spacing: 20) {
            if accountService.isSignedIn {
                signedInView
            } else {
                authFormView
            }
        }
        .padding(24)
        .frame(width: 360)
    }

    // MARK: - Signed In

    private var signedInView: some View {
        VStack(spacing: 16) {
            Image(systemName: "person.circle.fill")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)

            if let sub = accountService.subscription {
                VStack(spacing: 4) {
                    Text(sub.displayStatus)
                        .font(.headline)
                    if let tier = sub.tier {
                        Text("Engram \(tier.capitalized)")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }
            } else {
                Text("Free Tier")
                    .font(.headline)
                Text("Upgrade to Pro for cloud sync & backup")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Text("Sync Endpoint")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("https://api.engram.io", text: $accountService.endpoint)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
            }

            HStack {
                Button("Sign Out") {
                    accountService.signOut()
                }

                Spacer()

                Button("Refresh") {
                    Task { await accountService.refreshSubscriptionStatus() }
                }
            }
        }
    }

    // MARK: - Auth Form

    private var authFormView: some View {
        VStack(spacing: 16) {
            Text("Sign In")
                .font(.title2)
                .fontWeight(.semibold)

            Text("Sign in to enable cloud sync and backup.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            // OAuth buttons
            SignInWithAppleButton(.signIn) { request in
                request.requestedScopes = [.email, .fullName]
            } onCompletion: { result in
                Task { await accountService.handleAppleSignIn(result: result) }
            }
            .frame(height: 38)

            GoogleSignInButton {
                Task { await accountService.signInWithGoogle() }
            }
            .frame(height: 38)

            // Divider between OAuth and email/password
            HStack(spacing: 8) {
                Rectangle().frame(height: 1).foregroundStyle(.separator)
                Text("or")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Rectangle().frame(height: 1).foregroundStyle(.separator)
            }

            // Email/password form
            VStack(spacing: 10) {
                TextField("Email", text: $email)
                    .textFieldStyle(.roundedBorder)
                    .textContentType(.emailAddress)

                SecureField("Password", text: $password)
                    .textFieldStyle(.roundedBorder)
                    .textContentType(isRegistering ? .newPassword : .password)
            }

            if let error = accountService.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
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
            .disabled(email.isEmpty || password.isEmpty || accountService.isLoading)

            Button(isRegistering ? "Already have an account? Sign in" : "Don't have an account? Register") {
                isRegistering.toggle()
                accountService.errorMessage = nil
            }
            .font(.caption)
            .buttonStyle(.plain)
            .foregroundStyle(.blue)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Text("Server")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("https://api.engram.io", text: $accountService.endpoint)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
            }
        }
    }
}
