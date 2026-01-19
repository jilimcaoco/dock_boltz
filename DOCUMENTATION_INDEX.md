# Documentation Index

Complete guide to all documentation, tools, and examples.

## 📋 Getting Started

**Start here for quick overview:**
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) - Project status and quick start
- [BATCH_SCORING_TOOLKIT.md](BATCH_SCORING_TOOLKIT.md) - Main toolkit overview

**For immediate use:**
- [YAML_GENERATOR_QUICK_REF.md](YAML_GENERATOR_QUICK_REF.md) - Quick reference for YAML generator (5 min read)

## 🔧 Tools & Scripts

### YAML Generator
**Purpose:** Generate configuration files for batch scoring  
**Script:** [scripts/generate_affinity_yaml.py](scripts/generate_affinity_yaml.py)  
**Size:** ~400 lines, production-ready  

```bash
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir ./poses/ \
    --output batch_affinity.yaml
```

**Reference:** [YAML_GENERATOR_QUICK_REF.md](YAML_GENERATOR_QUICK_REF.md)

### Affinity Predictor
**Purpose:** Score structures with Boltz-2  
**Script:** `python -m boltz.main_simplified predict`  

```bash
python -m boltz.main_simplified predict \
    --yaml batch_affinity.yaml \
    --checkpoint checkpoint.ckpt \
    --output results/
```

### Test Suite
**Purpose:** Validate external structure loading  
**Script:** [test_external_structures.py](test_external_structures.py)  
**Size:** ~300 lines, includes 5 test categories  

```bash
python test_external_structures.py
```

## 📚 Documentation

### Core Documentation

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | Project status, architecture, quick start | 5 min |
| [BATCH_SCORING_TOOLKIT.md](BATCH_SCORING_TOOLKIT.md) | Toolkit overview, workflows, common tasks | 10 min |
| [YAML_GENERATOR_QUICK_REF.md](YAML_GENERATOR_QUICK_REF.md) | Quick reference for YAML generator | 5 min |
| [BATCH_AFFINITY_SCORING.md](BATCH_AFFINITY_SCORING.md) | Detailed workflow guide, examples | 20 min |
| [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) | Testing, validation, troubleshooting | 15 min |

### Technical Documentation

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [EXTERNAL_STRUCTURES_GUIDE.md](EXTERNAL_STRUCTURES_GUIDE.md) | How external structures work | 10 min |
| [AFFINITY_MODULE_GUIDE.md](AFFINITY_MODULE_GUIDE.md) | Affinity module architecture | 10 min |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | One-page reference card | 2 min |

### Code Documentation

| File | Lines | Purpose |
|------|-------|---------|
| [src/boltz/data/parse/schema.py](src/boltz/data/parse/schema.py) | ~1950 | YAML parser with structure_path support |
| [src/boltz/model/models/affinity_predictor.py](src/boltz/model/models/affinity_predictor.py) | ~280 | Simplified inference model |
| [src/boltz/main_simplified.py](src/boltz/main_simplified.py) | ~220 | CLI interface |

## 📝 Examples

### Complete Working Example
**Directory:** [examples/batch_scoring_example/](examples/batch_scoring_example/)  
**Files:** README.md + step-by-step commands  
**Time to run:** 15-30 min (depending on pose count)  

**Includes:**
- Directory structure reference
- Step-by-step workflow (Vina → Boltz)
- Score extraction and ranking
- Comparison with Vina scores

### Example YAML Configurations

| File | Purpose |
|------|---------|
| [examples/affinity.yaml](examples/affinity.yaml) | Basic affinity config |
| [examples/affinity_with_structures.yaml](examples/affinity_with_structures.yaml) | External structures example |

## 🧪 Testing Resources

### Automated Tests
- [test_external_structures.py](test_external_structures.py) - Unit tests for structure loading

**Tests include:**
1. YAML parsing with structure_path
2. Protein coordinate loading
3. Ligand coordinate loading
4. Coordinate shape validation
5. Full pipeline integration

### Manual Testing
- [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) - Comprehensive testing guide

**Covers:**
- Unit tests (YAML, structures, shapes)
- Integration tests (realistic workflows)
- Edge case testing (errors, mismatches)
- Production validation (real docking data)

## 📖 How to Read This Documentation

### Scenario 1: I have docked poses and want to score them
1. Read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md#quick-start) (2 min)
2. Read [YAML_GENERATOR_QUICK_REF.md](YAML_GENERATOR_QUICK_REF.md) (5 min)
3. Follow [examples/batch_scoring_example/README.md](examples/batch_scoring_example/README.md) (15 min)
4. Run your docking data through the pipeline

**Total time:** ~20 minutes

### Scenario 2: I want to understand the architecture
1. Read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md#whats-been-implemented) (5 min)
2. Read [EXTERNAL_STRUCTURES_GUIDE.md](EXTERNAL_STRUCTURES_GUIDE.md) (10 min)
3. Read [AFFINITY_MODULE_GUIDE.md](AFFINITY_MODULE_GUIDE.md) (10 min)
4. Review code in src/boltz/

**Total time:** ~30 minutes

### Scenario 3: I'm setting up the pipeline for the first time
1. Read [BATCH_SCORING_TOOLKIT.md](BATCH_SCORING_TOOLKIT.md) (10 min)
2. Run [test_external_structures.py](test_external_structures.py) (2 min)
3. Follow [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md#phase-2-integration-test) (15 min)
4. Test with sample data

**Total time:** ~30 minutes

### Scenario 4: I need to troubleshoot an issue
1. Check [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md#troubleshooting) - Troubleshooting section
2. Check [BATCH_AFFINITY_SCORING.md](BATCH_AFFINITY_SCORING.md#troubleshooting) - FAQ and troubleshooting
3. Run relevant tests from [test_external_structures.py](test_external_structures.py)
4. Review error messages and logs

**Total time:** ~15 minutes

## 🗺️ File Map

```
Documentation/
├── IMPLEMENTATION_STATUS.md        ← Start here
├── BATCH_SCORING_TOOLKIT.md       ← Main overview
├── YAML_GENERATOR_QUICK_REF.md    ← For YAML generator
├── BATCH_AFFINITY_SCORING.md      ← Detailed guide
├── TESTING_CHECKLIST.md           ← Testing & troubleshooting
├── EXTERNAL_STRUCTURES_GUIDE.md   ← Architecture
├── AFFINITY_MODULE_GUIDE.md       ← Module details
├── QUICK_REFERENCE.md             ← One-pager
├── DOCUMENTATION_INDEX.md         ← This file
│
Scripts/
├── scripts/generate_affinity_yaml.py  ← Main tool
├── test_external_structures.py       ← Test suite
│
Examples/
├── examples/batch_scoring_example/
│   └── README.md                     ← Full example
├── examples/affinity.yaml
└── examples/affinity_with_structures.yaml

Source Code/
├── src/boltz/data/parse/schema.py        ← Structure loading
├── src/boltz/model/models/affinity_predictor.py
└── src/boltz/main_simplified.py          ← CLI
```

## 📊 Quick Navigation by Topic

### Batch Processing
- [BATCH_SCORING_TOOLKIT.md](BATCH_SCORING_TOOLKIT.md)
- [YAML_GENERATOR_QUICK_REF.md](YAML_GENERATOR_QUICK_REF.md)
- [examples/batch_scoring_example/README.md](examples/batch_scoring_example/README.md)

### Docking Integration
- [BATCH_AFFINITY_SCORING.md](BATCH_AFFINITY_SCORING.md#integration-with-other-tools)
- [examples/batch_scoring_example/README.md](examples/batch_scoring_example/README.md)

### External Structures
- [EXTERNAL_STRUCTURES_GUIDE.md](EXTERNAL_STRUCTURES_GUIDE.md)
- [AFFINITY_MODULE_GUIDE.md](AFFINITY_MODULE_GUIDE.md)

### Testing & Validation
- [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md)
- [test_external_structures.py](test_external_structures.py)

### Troubleshooting
- [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md#troubleshooting)
- [BATCH_AFFINITY_SCORING.md](BATCH_AFFINITY_SCORING.md#troubleshooting)

### Performance & HPC
- [BATCH_AFFINITY_SCORING.md](BATCH_AFFINITY_SCORING.md#performance-tips)
- [examples/batch_scoring_example/README.md](examples/batch_scoring_example/README.md#parallel-gpu-processing)

### API & Integration
- [BATCH_SCORING_TOOLKIT.md](BATCH_SCORING_TOOLKIT.md#common-tasks)
- Code examples in each doc

## 📱 Mobile-Friendly Resources

**For quick reference on phone/tablet:**
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - One-page summary
- [YAML_GENERATOR_QUICK_REF.md](YAML_GENERATOR_QUICK_REF.md) - Compact reference

## 🔗 Important Links

### Checkpoint Download
- Boltz-2 models: https://huggingface.co/boltz-community/Boltz-2

### Related Tools
- AutoDock Vina: http://vina.scripps.edu/
- OpenBabel: http://openbabel.org/
- RDKit: https://www.rdkit.org/

## ✅ Verification Checklist

Before you start:
- [ ] Read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- [ ] Understand the pipeline from [BATCH_SCORING_TOOLKIT.md](BATCH_SCORING_TOOLKIT.md)
- [ ] Have Boltz-2 checkpoint downloaded
- [ ] Have docking software installed (Vina or other)
- [ ] Have structure files in correct format (PDB/MOL2/SDF)

## 📞 Getting Help

1. **Quick answer?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (2 min)
2. **How do I...?** → [BATCH_SCORING_TOOLKIT.md](BATCH_SCORING_TOOLKIT.md) (10 min)
3. **Issue/error?** → [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md#troubleshooting) (15 min)
4. **Deep dive?** → [BATCH_AFFINITY_SCORING.md](BATCH_AFFINITY_SCORING.md) (30 min)

---

**Last Updated:** January 2026  
**Status:** Complete ✓  
**Total Documentation:** 9 guides + code examples
