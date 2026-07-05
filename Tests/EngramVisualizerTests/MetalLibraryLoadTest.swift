import Testing
import Metal
import EngramMetalShaders

@Suite("Metal Library Loading")
struct MetalLibraryLoadTest {
    @Test("EngramMetalShaders.makeLibrary loads compiled metallib from Bundle.module")
    func testLoadLibrary() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let library = try EngramMetalShaders.makeLibrary(device: device)
        print("Functions: \(library.functionNames)")
        #expect(library.functionNames.contains("compute_charge_forces"))
        #expect(library.functionNames.contains("compute_charge_forces_bh"))
        #expect(library.functionNames.contains("compute_spring_forces"))
    }
}
