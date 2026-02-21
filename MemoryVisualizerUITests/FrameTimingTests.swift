import XCTest

@MainActor
final class FrameTimingTests: XCTestCase {
    let app = XCUIApplication()

    override func setUpWithError() throws {
        continueAfterFailure = false
        app.launch()
        sleep(5)
    }

    func testCapture3DFrameTiming() throws {
        // The 3D button has identifier "dimension-toggle" and label "3D"
        let threeDButton = app.buttons.matching(NSPredicate(format: "label == '3D'")).firstMatch
        XCTAssertTrue(threeDButton.waitForExistence(timeout: 10), "3D button not found")
        threeDButton.tap()
        sleep(3) // let 3D view initialize and settle

        // Use window as drag target since RealityView may not expose an element
        let window = app.windows.firstMatch
        XCTAssertTrue(window.exists)

        // Orbit: drag from center to various directions
        let center = window.coordinate(withNormalizedOffset: CGVector(dx: 0.4, dy: 0.5))
        let right = window.coordinate(withNormalizedOffset: CGVector(dx: 0.7, dy: 0.5))
        let left = window.coordinate(withNormalizedOffset: CGVector(dx: 0.1, dy: 0.5))
        let top = window.coordinate(withNormalizedOffset: CGVector(dx: 0.4, dy: 0.2))
        let bottom = window.coordinate(withNormalizedOffset: CGVector(dx: 0.4, dy: 0.8))

        // Slow orbit right
        center.press(forDuration: 0.1, thenDragTo: right, withVelocity: .slow, thenHoldForDuration: 0.1)
        sleep(2) // let camera settle

        // Slow orbit left
        center.press(forDuration: 0.1, thenDragTo: left, withVelocity: .slow, thenHoldForDuration: 0.1)
        sleep(2)

        // Slow orbit up
        center.press(forDuration: 0.1, thenDragTo: top, withVelocity: .slow, thenHoldForDuration: 0.1)
        sleep(2)

        // Slow orbit down
        center.press(forDuration: 0.1, thenDragTo: bottom, withVelocity: .slow, thenHoldForDuration: 0.1)
        sleep(2)

        // Fast orbit
        center.press(forDuration: 0.05, thenDragTo: right, withVelocity: .fast, thenHoldForDuration: 0.05)
        sleep(2)
    }
}
