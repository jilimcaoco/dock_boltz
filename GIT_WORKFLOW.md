# How to Save Your Fork on Git

You have made substantial changes to the dock_boltz project. Here's how to save them:

## Step 1: Check Current Status
```bash
git status
```
✓ Already done - shows 50+ modified/new files

## Step 2: Stage All Changes
```bash
# Add all modified and new files
git add .

# Or add specific categories:
git add scripts/generate_affinity_yaml.py  # YAML generator
git add src/boltz/model/models/affinity_predictor.py  # Affinity model
git add src/boltz/main_simplified.py  # Simplified CLI
git add src/boltz/data/parse/schema.py  # Structure loading
git add test_external_structures.py  # Test suite
git add *.md  # All documentation
```

## Step 3: Review Changes Before Committing
```bash
# See what will be committed
git diff --cached | head -50

# Or see summary
git status
```

## Step 4: Create Commit
```bash
# Commit with a descriptive message
git commit -m "feat: Add batch affinity scoring with external structure support

- Add YAML generator script for docked pose batch processing
- Implement external structure loading (PDB/MOL2/SDF)
- Add simplified affinity predictor model
- Auto-extract protein sequence from PDB files
- Comprehensive documentation and examples
- Test suite for validation"
```

## Step 5: Push to Your Fork
```bash
# First, set up remote if needed
git remote add origin https://github.com/YOUR_USERNAME/dock_boltz.git

# Push to your fork
git push -u origin main

# Or if main already exists:
git push origin main
```

## Step 6: Create Pull Request (Optional)
If you want to contribute back to the original project:
1. Go to https://github.com/YOUR_USERNAME/dock_boltz
2. Click "Pull Request"
3. Compare your fork with the original repository
4. Add description and submit

## Quick Reference Commands

```bash
# Add all changes
git add .

# Check what's staged
git status

# Commit
git commit -m "Your message"

# Push
git push origin main

# View commit history
git log --oneline -10

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Check what branch you're on
git branch -a
```

## File Organization (What's Being Saved)

### New Tools
- `scripts/generate_affinity_yaml.py` - YAML generator (400 lines)
- `test_external_structures.py` - Test suite (300 lines)
- `src/boltz/main_simplified.py` - Simplified CLI
- `src/boltz/model/models/affinity_predictor.py` - Affinity model

### Documentation (9 Files)
- `START_HERE.md` - Quick start guide
- `BATCH_SCORING_TOOLKIT.md` - Toolkit overview
- `YAML_GENERATOR_QUICK_REF.md` - Quick reference
- `BATCH_AFFINITY_SCORING.md` - Complete workflow
- `TESTING_CHECKLIST.md` - Testing guide
- Plus 4 more technical docs

### Modified Files
- `src/boltz/data/parse/schema.py` - Added structure_path support
- `README.md` - Updated

### Deleted Files
- Training scripts, confidence modules, diffusion models (cleanup)

## Configuration

Before your first commit, set your git identity:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Common Issues

### "fatal: not a git repository"
```bash
cd /Users/limcaoco/Projects/dock_boltz
git init
```

### "permission denied" when pushing
Check SSH key setup:
```bash
ssh -T git@github.com
```

### Large files
Git tracks file size. If you have large binaries, they may need LFS:
```bash
git lfs install
git lfs track "*.ckpt"
git add .gitattributes
```

## Next Steps

1. **Stage changes:**
   ```bash
   git add .
   ```

2. **Commit:**
   ```bash
   git commit -m "Add batch affinity scoring pipeline"
   ```

3. **Push:**
   ```bash
   git push -u origin main
   ```

Done! Your fork is now saved on GitHub.
