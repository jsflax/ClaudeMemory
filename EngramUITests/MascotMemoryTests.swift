import XCTest
import Lattice
import EngramKit

/// Simple visual test: seeds a small graph, launches the app, then inserts
/// a memory so you can watch the mascot's conjure animation in real time.
@MainActor
final class MascotMemoryTests: XCTestCase {
    let app = XCUIApplication()
    private var dbPath: String!
    private let testUUID = UUID().uuidString

    override func setUpWithError() throws {
        continueAfterFailure = false

        let tmpDir = NSTemporaryDirectory() + "mascot-test-\(testUUID)/"
        try FileManager.default.createDirectory(atPath: tmpDir, withIntermediateDirectories: true)
        dbPath = tmpDir + "memory.sqlite"

        // Seed a small galaxy so the mascot has nodes to patrol
        seedDatabase(at: dbPath, project: "Engram", memoryCount: 20, staleAge: 3600)

        // Point the app at our test DB
        app.launchEnvironment["CLAUDE_MEMORY_DB"] = dbPath
        app.launchEnvironment["ENGRAM_TEST_NO_NOTIFY"] = "1"
        app.launch()

        // Let the app load data and the mascot start patrolling
        sleep(6)
    }

    override func tearDownWithError() throws {
        app.terminate()
        let testDir = (dbPath as NSString).deletingLastPathComponent
        try? FileManager.default.removeItem(atPath: testDir)
    }

    func testMascotConjuresMemory() throws {
        let window = app.windows.firstMatch
        XCTAssertTrue(window.exists, "App window not found")

        // Defocus search bar so keys go to the 3D scene
        app.typeKey(.escape, modifierFlags: [])
        usleep(200_000)

        // Teleport camera to center of graph
        app.typeKey("t", modifierFlags: [])
        sleep(2)

        takeScreenshot(name: "mascot-before-insert")

        // Insert a memory — the xproc notification triggers
        // MascotFleet.onNodeCreated → mascot conjure animation
        print(">>> Inserting memory — watch the mascot!")
        insertMemories(at: dbPath, project: "Engram", count: 1)

        // Sleep so you can watch the conjure animation play out
        sleep(15)

        takeScreenshot(name: "mascot-after-conjure")

        // Insert a few more to see repeated reactions
        print(">>> Inserting 3 more memories…")
        for i in 0..<3 {
            insertMemories(at: dbPath, project: "Engram", count: 1)
            sleep(5)
            takeScreenshot(name: "mascot-conjure-\(i)")
        }

        // Final pause to observe
        sleep(10)
        takeScreenshot(name: "mascot-final")
    }
}
