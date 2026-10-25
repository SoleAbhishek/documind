  ### 2. Why Reduce Embedding Dimensions from 3072 to 768?                                                                                                     
                                                                                                                                                               
  By default, Google’s gemini-embedding-001 and gemini-embedding-2 produce 3,072-dimensional vectors. While saving storage is one benefit, the reasons for     
  reducing it to 768 or 1536 in production systems go much deeper:                                                                                             
  ──────                                                                                                                                                       
  #### Reason 1: The Curse of Dimensionality & Distance Metric Degradation                                                                                     
                                                                                                                                                               
  In high-dimensional space (D → ∞):                                                                                                                           
                                                                                                                                                               
        dist    - dist                                                                                                                                         
            max       min                                                                                                                                      
    lim ───────────────── → 0                                                                                                                                  
    D→∞      dist                                                                                                                                              
                 min                                                                                                                                           
                                                                                                                                                               
  • In 3,072 dimensions, points in hyperspace become increasingly equidistant from one another.                                                                
  • As the ratio between the nearest and farthest neighbor narrows, vector databases struggle with noisy nearest-neighbor boundaries during approximate nearest
  neighbor (ANN) graph traversal.                                                                                                                              
  • Truncating to 768 or 1536 dimensions often provides sharper contrast and better cluster separation for retrieval.                                          
  ──────                                                                                                                                                       
  #### Reason 2: HNSW Graph Search Latency & CPU Cache Throughput                                                                                              
                                                                                                                                                               
  Every search query performs hundreds of vector distance computations during HNSW (Hierarchical Navigable Small World) graph traversal:                       
                                                                                                                                                               
                        𝐮 · 𝐯                                                                                                                                  
    Cosine Similarity = ──────                                                                                                                                 
                        ‖𝐮‖‖𝐯‖                                                                                                                                 
                                                                                                                                                               
  • At 3,072 dimensions (Float32):                                                                                                                             
      • Each vector = 12,288 bytes (~12.3 KB).                                                                                                                 
      • Comparing 2 vectors requires streaming 12 KB from RAM into CPU L1/L2 cache and executing 3,072 floating-point multiplications.                         
  • At 768 dimensions (Float32):                                                                                                                               
      • Each vector = 3,072 bytes (~3.1 KB) — exactly 4x smaller.                                                                                              
      • A 768-dim vector fits neatly into CPU SIMD registers (AVX-512 / AVX2).                                                                                 
      • Distance computation executes in 1/4 the CPU instruction cycles, yielding 2.5x to 3.5x lower search latency under concurrent load.                     
                                                                                                                                                               
  ──────                                                                                                                                                       
  #### Reason 3: RAM Footprint in Vector Databases                                                                                                             
  
  Qdrant (and other vector DBs like Milvus or Pinecone) stores vectors and their HNSW graph edges in RAM to maintain sub-5ms retrieval speeds.
  
   Scale                             │ RAM at 3,072 dims                        │ RAM at 768 dims                          │ Difference
  ───────────────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────
   50,000 Chunks                     │ ~615 MB vectors + ~500 MB graph ≈ 1.1 GB │ ~150 MB vectors + ~250 MB graph ≈ 400 MB │ ~3x less RAM
   500,000 Chunks                    │ ~6.1 GB vectors + ~5 GB graph ≈ 11 GB    │ ~1.5 GB vectors + ~2.5 GB graph ≈ 4 GB   │ Can run on an 8GB machine vs 16GB
   5,000,000 Chunks                  │ ~61 GB vectors + ~50 GB graph ≈ 110 GB   │ ~15 GB vectors + ~25 GB graph ≈ 40 GB    │ ~$3,000/yr infrastructure savings
  ──────
  #### Reason 4: Matryoshka Representation Learning (MRL)
  
  Why does Google Gemini provide the output_dimensionality parameter?
  
  • Google trained gemini-embedding using Matryoshka Representation Learning (MRL), named after Russian nesting dolls 🪆.                                      
  • In MRL, the loss function explicitly forces the model to encode the most critical, high-variance semantic signals in the first N dimensions (d₁ … d₇₆₈),   
  with later dimensions encoding finer, diminishing-return nuances.
  • Benchmark evaluations on MTEB (Massive Text Embedding Benchmark) demonstrate:
      • 768 dimensions retains ~98.5% to 99% of the full 3,072 retrieval accuracy.
      • 1536 dimensions retains ~99.5%.
  • That means using 3,072 dimensions costs 400% more RAM and 3x more CPU latency for a ~1% difference in retrieval score.