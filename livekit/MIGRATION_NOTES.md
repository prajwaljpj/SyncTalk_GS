# Repository Reorganization - Migration Notes

## What Was Changed

All LiveKit-related files have been moved into a dedicated `livekit/` directory for better repository organization.

## Directory Structure Changes

### Before (Old Structure)
```
SyncTalk/
├── livekit_avatar/
│   ├── agent_worker.py
│   ├── avatar_worker.py
│   ├── synctalk_video_generator.py
│   └── ditto_video_generator.py
├── livekit_client/
│   ├── simple_client.html
│   └── token_server.py
├── livekit_server.sh
├── LIVEKIT_INTEGRATION_GUIDE.md
├── LIVEKIT_SETUP_COMPLETE.md
└── [other project files...]
```

### After (New Structure)
```
SyncTalk/
├── livekit/                    # ← New dedicated directory
│   ├── server/
│   │   ├── agent_worker.py
│   │   ├── avatar_worker.py
│   │   ├── synctalk_video_generator.py
│   │   └── ditto_video_generator.py
│   ├── client/
│   │   ├── simple_client.html
│   │   └── token_server.py
│   ├── docs/
│   │   ├── LIVEKIT_INTEGRATION_GUIDE.md
│   │   └── LIVEKIT_SETUP_COMPLETE.md
│   ├── livekit_server.sh
│   └── README.md              # ← New overview document
└── [other project files...]
```

## Files Modified

### 1. Code Files (Path Updates)

#### `livekit/livekit_server.sh`
- **Line 99**: Updated agent worker path
  ```bash
  # Old: uv run python livekit_avatar/agent_worker.py dev
  # New: uv run python livekit/server/agent_worker.py dev
  ```

#### `livekit/server/avatar_worker.py`
- **Lines 28-30**: Updated import paths
  ```python
  # Old: sys.path.insert(0, str(Path(__file__).parent))
  # New: sys.path.insert(0, str(Path(__file__).parent))  # livekit/server/
  #      sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root
  ```

#### `livekit/server/synctalk_video_generator.py`
- **Lines 28-30**: Updated import paths
  ```python
  # Old: sys.path.insert(0, str(Path(__file__).parent.parent))
  # New: sys.path.insert(0, str(Path(__file__).parent.parent.parent))
  ```

### 2. Documentation Files (Path References)

#### `livekit/docs/LIVEKIT_SETUP_COMPLETE.md`
Updated all file path references:
- `livekit_avatar/` → `livekit/server/`
- `livekit_client/` → `livekit/client/`
- `livekit_server.sh` → `livekit/livekit_server.sh`
- Added reference to `livekit/docs/LIVEKIT_INTEGRATION_GUIDE.md`

### 3. Project Files

#### `README.md` (Project Root)
- **Lines 214-235**: Added new "LiveKit Integration" section
  - Overview of real-time conversational avatar integration
  - Quick start instructions
  - Links to documentation in `livekit/docs/`

## New Files Created

1. **`livekit/README.md`** - Overview of the LiveKit integration
   - Directory structure explanation
   - Quick start guide
   - Configuration reference
   - Troubleshooting tips

2. **`livekit/MIGRATION_NOTES.md`** - This file

## Breaking Changes

### For Users

If you have existing scripts or commands that reference the old paths, update them:

```bash
# Old commands:
./livekit_server.sh
cd livekit_client && python token_server.py
python livekit_avatar/agent_worker.py dev

# New commands:
./livekit/livekit_server.sh
cd livekit/client && python token_server.py
python livekit/server/agent_worker.py dev
```

### For Developers

If you have local modifications to these files:

1. **Backup your changes** before pulling
2. **Update import paths** in your custom code:
   ```python
   # Old:
   from livekit_avatar.synctalk_video_generator import SyncTalkVideoGenerator

   # New:
   from livekit.server.synctalk_video_generator import SyncTalkVideoGenerator
   ```

3. **Update file references** in your scripts:
   ```bash
   # Old:
   CONFIG_FILE="livekit_avatar/config.yaml"

   # New:
   CONFIG_FILE="livekit/server/config.yaml"
   ```

## Migration Checklist

If you're upgrading from the old structure:

- [ ] Update any custom scripts that reference `livekit_avatar/` or `livekit_client/`
- [ ] Update bookmarks/documentation pointing to old file paths
- [ ] Update CI/CD pipelines if they reference old paths
- [ ] Test the startup script: `./livekit/livekit_server.sh`
- [ ] Test the client: `cd livekit/client && python token_server.py`

## Benefits of New Structure

1. **Cleaner Repository**: All LiveKit-related files in one place
2. **Better Organization**: Clear separation of server, client, and docs
3. **Easier Navigation**: Dedicated README for LiveKit integration
4. **Scalability**: Easier to add new LiveKit features
5. **Documentation**: Centralized docs in `livekit/docs/`

## Questions?

- See [livekit/README.md](README.md) for overview
- See [livekit/docs/LIVEKIT_SETUP_COMPLETE.md](docs/LIVEKIT_SETUP_COMPLETE.md) for setup
- See [livekit/docs/LIVEKIT_INTEGRATION_GUIDE.md](docs/LIVEKIT_INTEGRATION_GUIDE.md) for details

## Rollback (Not Recommended)

If you need to revert to the old structure (not recommended):

```bash
# Move files back to old locations
mv livekit/server/* livekit_avatar/
mv livekit/client/* livekit_client/
mv livekit/livekit_server.sh ./
mv livekit/docs/*.md ./

# Remove new directory
rm -rf livekit/

# Restore old paths in code (reverse the changes above)
```

**Note:** This is not recommended as the new structure is cleaner and more maintainable.
