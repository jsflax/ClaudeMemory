import AppKit
import AuthenticationServices
import Foundation
import GoogleSignIn
import Observation

/// Manages authentication and subscription state for the Engram cloud service.
@Observable
@MainActor
final class AccountService {
    private static let tokenKey = "auth_token"
    private static let endpointKey = "sync_endpoint"
    private static let subscriptionStatusKey = "subscription_status"
    private static let subscriptionTierKey = "subscription_tier"

    /// The sync server endpoint. Defaults to production.
    var endpoint: String {
        didSet { UserDefaults.standard.set(endpoint, forKey: Self.endpointKey) }
    }

    /// Current auth token, persisted in Keychain.
    private(set) var token: String?

    /// Whether the user is signed in.
    var isSignedIn: Bool { token != nil }

    /// Current subscription status. Persisted to UserDefaults so sync can auto-connect at launch.
    private(set) var subscription: SubscriptionInfo? {
        didSet {
            if let sub = subscription {
                UserDefaults.standard.set(sub.status, forKey: Self.subscriptionStatusKey)
                UserDefaults.standard.set(sub.tier, forKey: Self.subscriptionTierKey)
            } else {
                UserDefaults.standard.removeObject(forKey: Self.subscriptionStatusKey)
                UserDefaults.standard.removeObject(forKey: Self.subscriptionTierKey)
            }
        }
    }

    /// User profile info from the server.
    private(set) var userProfile: UserProfile?

    /// Whether a network operation is in progress.
    private(set) var isLoading = false

    /// Last error message for display.
    var errorMessage: String?

    init() {
        // Must match the sync daemon's default (EngramDaemon → engramdb.io);
        // the old engram.io default sent auth to a different host than the
        // daemon relayed sync to.
        self.endpoint = UserDefaults.standard.string(forKey: Self.endpointKey)
            ?? "https://engramdb.io"
        #if TEST_AUTH_TOKEN
        self.token = ProcessInfo.processInfo.environment["TEST_AUTH_TOKEN"]
            ?? KeychainHelper.load(for: Self.tokenKey)
        #else
        self.token = KeychainHelper.load(for: Self.tokenKey)
        #endif

        // Restore cached subscription so syncWebSocketURL is available synchronously at launch
        if let status = UserDefaults.standard.string(forKey: Self.subscriptionStatusKey) {
            self.subscription = SubscriptionInfo(
                status: status,
                tier: UserDefaults.standard.string(forKey: Self.subscriptionTierKey),
                currentPeriodEnd: nil
            )
        }

        if token != nil {
            Task {
                await fetchProfile()
                await refreshSubscriptionStatus()
            }
        }
    }

    // MARK: - Auth

    func signIn(email: String, password: String) async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        do {
            let url = URL(string: "\(endpoint)/login")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            // Server's LoginRequest decodes `email` (not `username`) — the
            // old key silently produced a decode failure / failed login.
            request.httpBody = try JSONEncoder().encode(["email": email, "password": password])

            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                errorMessage = "Invalid email or password."
                return
            }

            let loginResponse = try JSONDecoder().decode(LoginResponse.self, from: data)
            token = loginResponse.token
            userProfile = loginResponse.user
            try KeychainHelper.save(loginResponse.token, for: Self.tokenKey)
            await refreshSubscriptionStatus()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func register(email: String, password: String) async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        do {
            let url = URL(string: "\(endpoint)/register")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONEncoder().encode([
                "email": email,
                "username": email,
                "password": password,
            ])

            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                let body = String(data: data, encoding: .utf8) ?? ""
                errorMessage = "Registration failed: \(body)"
                return
            }

            // Auto-login after registration
            let loginResponse = try JSONDecoder().decode(LoginResponse.self, from: data)
            token = loginResponse.token
            try KeychainHelper.save(loginResponse.token, for: Self.tokenKey)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func signOut() {
        token = nil
        subscription = nil  // didSet clears UserDefaults
        userProfile = nil
        KeychainHelper.delete(for: Self.tokenKey)
    }

    // MARK: - Apple Sign In

    func handleAppleSignIn(result: Result<ASAuthorization, any Error>) async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        do {
            let auth = try result.get()
            guard let credential = auth.credential as? ASAuthorizationAppleIDCredential,
                  let identityTokenData = credential.identityToken,
                  let authCodeData = credential.authorizationCode,
                  let identityToken = String(data: identityTokenData, encoding: .utf8),
                  let authorizationCode = String(data: authCodeData, encoding: .utf8)
            else {
                errorMessage = "Apple sign-in: missing credentials."
                return
            }

            try await postAuthRequest(
                path: "/auth/apple",
                body: ["identityToken": identityToken, "authorizationCode": authorizationCode]
            )
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Google Sign In

    func signInWithGoogle() async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        guard GIDSignIn.sharedInstance.configuration != nil else {
            errorMessage = "Google Sign-In is not configured."
            return
        }

        do {
            guard let window = NSApp.keyWindow else {
                errorMessage = "No window available."
                return
            }

            let result = try await GIDSignIn.sharedInstance.signIn(withPresenting: window)
            guard let idToken = result.user.idToken?.tokenString else {
                errorMessage = "Google sign-in: missing ID token."
                return
            }

            try await postAuthRequest(
                path: "/auth/google",
                body: ["idToken": idToken]
            )
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Shared Auth

    private func postAuthRequest(path: String, body: [String: String]) async throws {
        let url = URL(string: "\(endpoint)\(path)")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "Unknown error"
            errorMessage = "Authentication failed: \(body)"
            return
        }

        let loginResponse = try JSONDecoder().decode(LoginResponse.self, from: data)
        token = loginResponse.token
        userProfile = loginResponse.user
        try KeychainHelper.save(loginResponse.token, for: Self.tokenKey)
//        writeSyncCredentialFile(token: loginResponse.token)
        await refreshSubscriptionStatus()
    }

    // MARK: - Profile

    func fetchProfile() async {
        guard let token else { return }

        do {
            let url = URL(string: "\(endpoint)/me")!
            var request = URLRequest(url: url)
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return }

            userProfile = try JSONDecoder().decode(UserProfile.self, from: data)
        } catch {
            // Non-fatal — profile fetch can fail silently
        }
    }

    // MARK: - Subscription

    func refreshSubscriptionStatus() async {
        guard let token else { return }

        do {
            let url = URL(string: "\(endpoint)/subscriptions/status")!
            var request = URLRequest(url: url)
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return }

            let info = try JSONDecoder().decode(SubscriptionInfo.self, from: data)
            subscription = info
        } catch {
            // Non-fatal — subscription check can fail silently
        }
    }

    /// Returns the WebSocket sync URL if the user has an active subscription.
    var syncWebSocketURL: URL? {
        guard isSignedIn, subscription?.isActive == true else { return nil }
        let wsEndpoint = endpoint
            .replacingOccurrences(of: "https://", with: "wss://")
            .replacingOccurrences(of: "http://", with: "ws://")
        return URL(string: "\(wsEndpoint)/sync")
    }

    // MARK: - Types

    struct LoginResponse: Decodable {
        let token: String
        let user: UserProfile?
    }

    struct UserProfile: Decodable {
        let id: UUID
        let email: String?
        let fullName: String?
        let profilePictureUrl: String?
        let providers: [String]?
    }

    struct SubscriptionInfo: Decodable {
        let status: String
        let tier: String?
        let currentPeriodEnd: Date?

        var isActive: Bool {
            status == "active" || status == "trialing"
        }

        var displayStatus: String {
            switch status {
            case "active": return "Active"
            case "trialing": return "Trial"
            case "past_due": return "Past Due"
            case "cancelled": return "Cancelled"
            case "none": return "Free"
            default: return status.capitalized
            }
        }

        enum CodingKeys: String, CodingKey {
            case status, tier
            case currentPeriodEnd = "current_period_end"
        }
    }
}
