# Interaction Node Schema

This document describes the standard API format for AI model interaction applications. Use this schema as a reference when creating new applications with different AI models and inputs.

## Process Status

All applications should track and return a process status that indicates the current state of the system:

```typescript
export const PROCESS_STATUS = {
  UNINITIALIZED: "UNINITIALIZED",
  INITIALIZED: "INITIALIZED",
  RUNNING: "RUNNING",
  PAUSED: "PAUSED",
  STOPPED: "STOPPED",
};
```

### Status Transitions

- **UNINITIALIZED** → **INITIALIZED**: After successful `/initialize` call
- **INITIALIZED** → **RUNNING**: After successful `/run` call
- **RUNNING** → **STOPPED**: After successful `/stop` call
- **RUNNING** → **PAUSED**: After pause operation (if implemented)
- **PAUSED** → **RUNNING**: After resume operation (if implemented)

## Standard Response Format

All API routes must return responses in the following format:

```json
{
  "status": "success" | "error",
  "message": "Human-readable message describing the result",
  "processStatus": "UNINITIALIZED" | "INITIALIZED" | "RUNNING" | "PAUSED" | "STOPPED",
  "optionalMessageData": {
    // Optional additional data specific to the route
    // Can be null if no additional data is needed
  }
}
```

### Response Fields

- **status** (required): Either `"success"` or `"error"` indicating the operation result
- **message** (required): A human-readable string describing what happened
- **processStatus** (required): Current process status from PROCESS_STATUS enum
- **optionalMessageData** (required): An object containing route-specific data, or `null` if not applicable

## Core Routes

### 1. Initialize Route

**Endpoint:** `POST /initialize`

**Purpose:** Initialize all algorithms, models, and input sources (cameras, microphones, etc.)

**Response Format:**

```json
{
  "status": "success",
  "message": "System initialized successfully",
  "processStatus": "INITIALIZED",
  "optionalMessageData": {
    "camera_ready": true,
    "models_loaded": true
    // Additional initialization details
  }
}
```

**Error Response:**

```json
{
  "status": "error",
  "message": "Initialization failed",
  "processStatus": "UNINITIALIZED",
  "optionalMessageData": {
    "camera_ready": false,
    "models_loaded": false
    // Error details
  }
}
```

**Status Updates:**

- Sets `processStatus` to `INITIALIZED` on success
- Keeps `processStatus` as `UNINITIALIZED` on failure

### 2. Run Route

**Endpoint:** `POST /run`

**Purpose:** Start the main processing loop (emotion detection, audio processing, etc.)

**Prerequisites:** System must be initialized (`processStatus` must be `INITIALIZED`)

**Response Format:**

```json
{
  "status": "success",
  "message": "Continuous emotion detection started",
  "processStatus": "RUNNING",
  "optionalMessageData": null
}
```

**Error Response (Not Initialized):**

```json
{
  "status": "error",
  "message": "System not initialized. Please call /initialize first.",
  "processStatus": "UNINITIALIZED",
  "optionalMessageData": null
}
```

**Error Response (Already Running):**

```json
{
  "status": "error",
  "message": "Emotion detection already running",
  "processStatus": "RUNNING",
  "optionalMessageData": null
}
```

**Status Updates:**

- Sets `processStatus` to `RUNNING` on success
- Returns current `processStatus` on error

### 3. Stop Route

**Endpoint:** `POST /stop`

**Purpose:** Stop the main processing loop

**Response Format:**

```json
{
  "status": "success",
  "message": "Emotion detection stopped",
  "processStatus": "STOPPED",
  "optionalMessageData": null
}
```

**Status Updates:**

- Sets `processStatus` to `STOPPED` on success

### 4. Video Feed Route (if applicable)

**Endpoint:** `GET /video_feed`

**Purpose:** Get the latest frame from the camera or video input

**Response Format:**

- Success: Returns JPEG image with `Content-Type: image/jpeg`
- Error: Returns JSON with standard error format

**Error Response:**

```json
{
  "status": "error",
  "message": "Failed to read frame from camera",
  "processStatus": "CURRENT_STATUS",
  "optionalMessageData": null
}
```

## Implementation Guidelines

### 1. State Management

- Maintain a global `process_status` variable that tracks the current state
- Update `process_status` immediately when state changes occur
- Always include `processStatus` in API responses

### 2. Error Handling

- All routes should use try-catch blocks
- Return appropriate HTTP status codes:
  - `200` for success
  - `400` for client errors (e.g., not initialized)
  - `500` for server errors
- Always include `processStatus` in error responses

### 3. Initialization

- The `/initialize` route should:
  - Load all required models/algorithms
  - Initialize all input sources (cameras, microphones, etc.)
  - Verify that everything is ready
  - Return detailed status in `optionalMessageData`

### 4. Process Control

- `/run` should only work if system is `INITIALIZED`
- `/stop` should work from any running state (`RUNNING` or `PAUSED`)
- Always update `processStatus` when starting or stopping

### 5. Optional Message Data

- Use `optionalMessageData` for route-specific information
- Set to `null` if no additional data is needed
- Common uses:
  - Initialization details (camera status, model status)
  - Current processing metrics
  - Error details beyond the message

## Example Application Structure

```python
# Process status constants
PROCESS_STATUS = {
    'UNINITIALIZED': 'UNINITIALIZED',
    'INITIALIZED': 'INITIALIZED',
    'RUNNING': 'RUNNING',
    'PAUSED': 'PAUSED',
    'STOPPED': 'STOPPED'
}

# Global state
process_status = PROCESS_STATUS['UNINITIALIZED']

@app.route('/initialize', methods=['POST'])
def initialize():
    global process_status
    try:
        # Initialize models and inputs
        # ...
        process_status = PROCESS_STATUS['INITIALIZED']
        return jsonify({
            'status': 'success',
            'message': 'System initialized successfully',
            'processStatus': process_status,
            'optionalMessageData': {
                # Initialization details
            }
        })
    except Exception as e:
        process_status = PROCESS_STATUS['UNINITIALIZED']
        return jsonify({
            'status': 'error',
            'message': f'Initialization error: {str(e)}',
            'processStatus': process_status,
            'optionalMessageData': None
        }), 500

@app.route('/run', methods=['POST'])
def run():
    global process_status
    if process_status != PROCESS_STATUS['INITIALIZED']:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized',
            'processStatus': process_status,
            'optionalMessageData': None
        }), 400

    # Start processing
    # ...
    process_status = PROCESS_STATUS['RUNNING']
    return jsonify({
        'status': 'success',
        'message': 'Processing started',
        'processStatus': process_status,
        'optionalMessageData': None
    })

@app.route('/stop', methods=['POST'])
def stop():
    global process_status
    # Stop processing
    # ...
    process_status = PROCESS_STATUS['STOPPED']
    return jsonify({
        'status': 'success',
        'message': 'Processing stopped',
        'processStatus': process_status,
        'optionalMessageData': None
    })
```

## Extending for Different AI Models

When creating applications for different AI models:

1. **Keep the same route structure**: `/initialize`, `/run`, `/stop`
2. **Maintain process status tracking**: Always update and return `processStatus`
3. **Follow response format**: Always return `status`, `message`, `processStatus`, `optionalMessageData`
4. **Customize optionalMessageData**: Include model-specific information (e.g., model name, confidence scores, etc.)
5. **Add model-specific routes**: If needed, add additional routes but maintain the standard format

## Notes

- This schema ensures consistency across different AI interaction applications
- The process status provides clear state tracking for frontend applications
- The standardized response format makes it easy to build generic clients
- All routes should be idempotent where possible (calling `/stop` multiple times should be safe)
