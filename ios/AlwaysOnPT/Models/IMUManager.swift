import Foundation
import CoreMotion
import Combine

@MainActor
final class IMUManager: ObservableObject {

    @Published var latestIMU: IMUData?
    @Published var isRunning = false

    private let motionManager = CMMotionManager()
    private let operationQueue = OperationQueue()
    private var updateInterval: TimeInterval

    init(frequencyHz: Double = 100) {
        self.updateInterval = 1.0 / frequencyHz
        operationQueue.name = "com.alwaysonpt.imu"
        operationQueue.maxConcurrentOperationCount = 1
    }

    func start() {
        guard !isRunning else { return }
        guard motionManager.isAccelerometerAvailable, motionManager.isGyroAvailable else { return }

        motionManager.deviceMotionUpdateInterval = updateInterval

        motionManager.startDeviceMotionUpdates(to: operationQueue) { [weak self] motion, error in
            guard let motion, error == nil else { return }

            let imu = IMUData(
                accelX: motion.userAcceleration.x + motion.gravity.x,
                accelY: motion.userAcceleration.y + motion.gravity.y,
                accelZ: motion.userAcceleration.z + motion.gravity.z,
                gyroX: motion.rotationRate.x,
                gyroY: motion.rotationRate.y,
                gyroZ: motion.rotationRate.z
            )

            Task { @MainActor [weak self] in
                self?.latestIMU = imu
            }
        }

        isRunning = true
    }

    func stop() {
        motionManager.stopDeviceMotionUpdates()
        isRunning = false
        latestIMU = nil
    }

    func setFrequency(_ hz: Double) {
        updateInterval = 1.0 / hz
        motionManager.deviceMotionUpdateInterval = updateInterval
    }
}
