# Implementation Plan: AI-Powered DJ Mixing Platform

**Branch**: `001-prototypes` | **Date**: 2025-09-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-prototypes/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, or `GEMINI.md` for Gemini CLI).
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Build an AI-powered DJ mixing platform that analyzes songs at the instrumental level using stem separation to create a 'Musical DNA' system. The platform enables intelligent DJ mixing with harmonic compatibility, automatic remix generation, and recommendations based on actual musical content rather than metadata. Key technical approach involves server-side processing with Demucs for stem separation, PostgreSQL with pgvector for vector storage, PyTorch Geometric for GNN recommendations, and React frontend with WebSocket real-time updates.

## Technical Context
The application uses Python FastAPI backend with PostgreSQL for storing audio metadata and musical features. Implement Demucs for stem separation using pre-trained models on GPU-enabled servers. Use Librosa for audio feature extraction and Essentia for beat/tempo detection. Store computed Musical DNA vectors in a vector database using pgvector extension. Implement the Camelot Wheel harmonic mixing logic in pure Python. Use PyTorch Geometric for the graph neural network recommendation system. Frontend built with React and Web Audio API for real-time audio visualization and mixing controls. Cache processed stems and features in Redis for fast retrieval. Use FAISS library for efficient similarity search across stem vectors. Deploy using Docker containers with separate services for API, stem processing workers, and recommendation engine. Store audio files locally or in S3-compatible storage with CDN for streaming. Implement WebSocket connections for real-time mixing updates. Use Celery with RabbitMQ for async processing tasks. All heavy computation happens server-side with results cached, keeping the client lightweight.

**Language/Version**: Python 3.11 (backend/ML), Node.js 18+ (dev tools), TypeScript 5.0+ (frontend)
**Primary Dependencies**: FastAPI, Demucs, PyTorch Geometric, librosa, Essentia, React, Web Audio API
**Storage**: PostgreSQL + pgvector (vectors), Redis (cache), S3/local (audio files), RabbitMQ (queues)
**Testing**: pytest (Python), Jest (TypeScript), Cypress (E2E)
**Target Platform**: Linux server (GPU-enabled), Modern web browsers
**Project Type**: web - distributed microservices architecture
**Performance Goals**: <2s song processing, <100ms mixing latency, 1000+ req/s API throughput
**Constraints**: GPU acceleration required, server-side heavy computation, lightweight client
**Scale/Scope**: 1M+ songs, 10k+ concurrent users, real-time WebSocket connections

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 3 (api-service, stem-processing-service, frontend-app)
- Using framework directly? Yes (FastAPI, React, PyTorch Geometric directly)
- Single data model? Yes (Musical DNA schema across services)
- Avoiding patterns? Yes (direct ORM usage, no unnecessary abstractions)

**Architecture**:
- EVERY feature as library? Yes (audio-processing, recommendation-engine, mixing-engine, web-client)
- Libraries listed:
  - audio-processing: Demucs stem separation, librosa/Essentia feature extraction, Musical DNA generation
  - recommendation-engine: PyTorch Geometric GNN, FAISS similarity search, Camelot Wheel logic
  - mixing-engine: real-time mixing logic, WebSocket handlers, cue point detection
  - web-client: React components, Web Audio API integration, waveform visualization
- CLI per library: Yes (--help, --version, --format json/text)
- Library docs: llms.txt format planned? Yes

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? Yes (tests written first, must fail, then implement)
- Git commits show tests before implementation? Yes (pre-commit hooks enforce)
- Order: Contract→Integration→E2E→Unit strictly followed? Yes
- Real dependencies used? Yes (actual PostgreSQL, Redis, RabbitMQ, GPU models)
- Integration tests for: ML model APIs, audio processing pipeline, WebSocket connections
- FORBIDDEN: Implementation before test, skipping RED phase - STRICTLY ENFORCED

**Observability**:
- Structured logging included? Yes (JSON format with request/session IDs)
- Frontend logs → backend? Yes (WebSocket log streaming)
- Error context sufficient? Yes (full audio processing pipeline tracing)

**Versioning**:
- Version number assigned? 0.1.0 (MAJOR.MINOR.BUILD)
- BUILD increments on every change? Yes (automated CI/CD)
- Breaking changes handled? Yes (API versioning, ML model backward compatibility)

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure]
```

**Structure Decision**: Option 2 - Web application (distributed microservices with React frontend)

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `/scripts/update-agent-context.sh [claude|gemini|copilot]` for your AI assistant
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P] 
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:
- TDD order: Tests before implementation 
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) - updated with specific tech choices
- [x] Phase 1: Design complete (/plan command) - refined for microservices + pgvector
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved - detailed tech specifications provided
- [x] Complexity deviations documented (none required)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*

## NEW UPDATE: Corrected Intelligence Architecture (2025-09-11)

### **Critical CLAP Integration Fixes**
Based on detailed review, the CLAP implementation needs major corrections:

**Corrected CLAP Usage**:
- Using **Microsoft CLAP or LAION CLAP** for audio-text embeddings (NOT the plugin API)
- **512-dimensional joint embeddings** instead of basic concatenation
- Pre-trained on 4M+ audio-text pairs with music-specific training
- Enables **zero-shot classification** without genre labels
- Latest **T-CLAP** adds temporal understanding for event ordering

**Enhanced Implementation**:
```python
from msclap import CLAP

class SemanticAudioEncoder:
    def __init__(self):
        self.clap = CLAP(version='music_audioset_epoch_15', use_cuda=True)
        
    def encode_audio(self, audio_path):
        return self.clap.get_audio_embeddings([audio_path])  # [1, 512]
        
    def compute_similarity(self, audio_embed, text_embed):
        return self.clap.compute_similarity(audio_embed, text_embed)
```

### **Hierarchical Intelligence Architecture Correction**
Replace basic feature extraction with three-tier intelligence:

**Layer 1 - Signal Intelligence (Enhanced)**:
- Hybrid Demucs for stem separation
- Multi-scale feature extraction with **proper STFT windowing**
- TCN-based beat tracking for improved temporal understanding

**Layer 2 - Semantic Intelligence (NEW)**:
- **CLAP embeddings** for cross-modal audio-text understanding  
- **SpecTNT** (91% accuracy) for structure detection vs 67% older methods
- **Valence-Arousal** neural networks for mood classification

**Layer 3 - Contextual Intelligence (ADVANCED)**:
- **DJ Context Transformer** for mix-point detection
- **Crowd Response LSTM** for energy prediction
- **Harmonic Compatibility GNN** for intelligent pairing

### **Advanced DJ-Specific Intelligence**
```python
class DJMixIntelligence:
    def __init__(self):
        self.transition_model = TransitionQualityNet()
        self.energy_flow_model = EnergyFlowLSTM()
        self.harmonic_path_planner = HarmonicA3C()  # Reinforcement learning
        
    def predict_mix_quality(self, track_a, track_b):
        # Neural prediction of transition quality
        # Trained on 1M+ real DJ transitions
        return quality_score  # 0-1
```

### **Multi-Resolution STFT Enhancement**
```python
def enhanced_stft(audio, sr=44100):
    # Proper Hann window with multiple resolutions
    stfts = []
    for n_fft in [512, 1024, 2048, 4096]:
        window = torch.hann_window(n_fft)
        stft = torch.stft(
            audio, 
            n_fft=n_fft,
            hop_length=n_fft//4,  # 25% overlap
            window=window,
            return_complex=True
        )
        stfts.append(stft)
    return stfts  # Multiple time-frequency resolutions
```

### **Training Pipeline Strategy**
1. **Self-supervised pretraining** (1M unlabeled tracks)
2. **Supervised finetuning** (100k labeled tracks) 
3. **Contrastive learning** (DJ mix transition data)
4. **Reinforcement learning** (simulated crowd response)

### **Intelligence Validation Metrics**
- Cue point detection: **F1 score > 0.9**
- Structure detection: **91% boundary precision/recall**
- Mix quality: **A/B testing with real DJs**
- Energy prediction: **Correlation with actual crowd movement**

This correction moves from basic feature matching to true **semantic musical understanding** trained on millions of real DJ decisions and audio-text pairs.

## UPDATED: Comprehensive Feature Extraction & Embedding Strategy (2025-09-12)

### **Maximum Feature Extraction Approach**
Instead of selective feature extraction, extract EVERYTHING possible first, then use feature selection:

**Audio Features (1000+ dimensions)**:
- Signal: STFT, MFCC, chroma, tonnetz, spectral features (centroid, rolloff, bandwidth, contrast)
- Rhythm: Beat tracking, onset detection, tempo variations, rhythmic patterns, syncopation measures
- Harmony: Key detection, chord progressions, harmonic network features, tonal stability
- Timbre: MFCCs, spectral shape, brightness, roughness, attack/decay characteristics
- Energy: RMS, peak levels, dynamic range, loudness (LUFS), energy distribution curves
- Structure: Segment boundaries, verse/chorus detection, repetition patterns

**Metadata Features (500+ dimensions)**:
- **Artist Information**: Main artist, featuring artists, producer, songwriter, label, country of origin
- **Geographic Data**: Chart positions by country, streaming popularity by region, cultural context
- **Genre Classification**: Primary genre, subgenres, fusion elements, decade/era classification  
- **Release Information**: Release date, album type, track position, duration, explicit content
- **Audio Quality**: Bitrate, sample rate, format, mastering characteristics, remaster status
- **Social Metrics**: Play counts, likes, shares, playlist inclusions across platforms
- **Acoustic Tags**: Mood descriptors, energy level, danceability, valence, instrumentalness

**Derived Features (300+ dimensions)**:
- **Cross-platform correlations**: Spotify vs YouTube vs SoundCloud metrics
- **Temporal patterns**: Seasonal popularity, time-of-day listening patterns
- **Network effects**: Similar artists, collaborative networks, influence graphs
- **Linguistic analysis**: Lyrics sentiment, language detection, topic modeling

### **Feature Selection & Dimensionality Reduction Pipeline**

**Stage 1: Statistical Filtering**
```python
class FeatureSelector:
    def filter_by_variance(self, features, threshold=0.01):
        # Remove features with near-zero variance
        return features[features.var() > threshold]
    
    def filter_by_correlation(self, features, threshold=0.95):
        # Remove highly correlated features
        correlation_matrix = features.corr()
        return self.remove_high_correlation(features, correlation_matrix, threshold)
```

**Stage 2: Musical Relevance Scoring**
```python
class MusicalRelevanceScorer:
    def score_dj_relevance(self, feature_name):
        # Score based on actual DJ decision making
        dj_importance_weights = {
            'tempo': 0.9, 'key': 0.9, 'energy': 0.8,
            'genre': 0.7, 'mood': 0.6, 'popularity': 0.4
        }
        return dj_importance_weights.get(feature_name, 0.1)
```

**Stage 3: Embedding Architecture**
```python
class ComprehensiveEmbedding:
    def __init__(self):
        # Multi-head attention for different feature types
        self.audio_encoder = AudioFeatureEncoder(input_dim=1000, output_dim=128)
        self.metadata_encoder = MetadataEncoder(input_dim=500, output_dim=64)  
        self.derived_encoder = DerivedFeatureEncoder(input_dim=300, output_dim=64)
        
        # Fusion layer
        self.fusion_layer = nn.Linear(256, 512)  # Final embedding dimension
        
    def forward(self, audio_features, metadata, derived_features):
        audio_emb = self.audio_encoder(audio_features)
        meta_emb = self.metadata_encoder(metadata)
        derived_emb = self.derived_encoder(derived_features)
        
        # Concatenate and fuse
        combined = torch.cat([audio_emb, meta_emb, derived_emb], dim=1)
        return self.fusion_layer(combined)
```

### **Natural Training Data Sources**

**Relationship-based Training (No Manual Labeling)**:
- **Same artist/album**: Style consistency embeddings should be close
- **Cover versions**: Different artists, same song → core musical similarity  
- **Remixes**: Original vs remix → structural similarity with variation
- **Genre clusters**: Songs in same subgenre should cluster together
- **Key compatibility**: Camelot wheel adjacent keys should be closer
- **BPM compatibility**: ±5% tempo tracks should be mixable
- **Chart co-occurrence**: Songs popular in same region/time should relate
- **Playlist co-occurrence**: Songs appearing together in curated playlists

**Contrastive Learning Objectives**:
```python
class MusicSimilarityLoss:
    def __init__(self):
        self.temperature = 0.1
        
    def forward(self, embeddings, similarity_labels):
        # InfoNCE loss for contrastive learning
        # Positive pairs: same artist, covers, remixes, key-compatible
        # Negative pairs: different genre, incompatible keys, different eras
        return info_nce_loss(embeddings, similarity_labels, self.temperature)
```

### **Implementation Priority**

**Phase 1**: Maximum feature extraction - implement comprehensive FeatureExtractor
**Phase 2**: Feature analysis - correlation, variance, relevance scoring  
**Phase 3**: Dimensionality reduction - PCA, feature selection, importance ranking
**Phase 4**: Embedding architecture - multi-encoder fusion model
**Phase 5**: Training pipeline - contrastive learning on natural relationships