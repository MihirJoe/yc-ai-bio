import Foundation

struct IMUData: Codable {
    let accelX: Double
    let accelY: Double
    let accelZ: Double
    let gyroX: Double
    let gyroY: Double
    let gyroZ: Double

    enum CodingKeys: String, CodingKey {
        case accelX = "accel_x"
        case accelY = "accel_y"
        case accelZ = "accel_z"
        case gyroX = "gyro_x"
        case gyroY = "gyro_y"
        case gyroZ = "gyro_z"
    }

    var accelMagnitude: Double {
        sqrt(accelX * accelX + accelY * accelY + accelZ * accelZ)
    }

    var gyroMagnitude: Double {
        sqrt(gyroX * gyroX + gyroY * gyroY + gyroZ * gyroZ)
    }
}

struct SensorDataMessage: Codable {
    let type: String
    let timestampMs: Int64
    let emg: [Double]
    let imu: IMUData?

    enum CodingKeys: String, CodingKey {
        case type
        case timestampMs = "timestamp_ms"
        case emg
        case imu
    }

    init(emg: [Double], imu: IMUData?, timestampMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000)) {
        self.type = "sensor_data"
        self.timestampMs = timestampMs
        self.emg = emg
        self.imu = imu
    }
}

struct SessionStartMessage: Codable {
    let type: String
    let taskType: String
    let sessionId: String

    enum CodingKeys: String, CodingKey {
        case type
        case taskType = "task_type"
        case sessionId = "session_id"
    }

    init(taskType: String, sessionId: String) {
        self.type = "session_start"
        self.taskType = taskType
        self.sessionId = sessionId
    }
}

struct DeepAnalysisRequestMessage: Codable {
    let type: String
    let taskType: String
    let question: String?

    enum CodingKeys: String, CodingKey {
        case type
        case taskType = "task_type"
        case question
    }

    init(taskType: String, question: String? = nil) {
        self.type = "deep_analysis_request"
        self.taskType = taskType
        self.question = question
    }
}

struct SessionEndMessage: Codable {
    let type: String

    init() {
        self.type = "session_end"
    }
}
