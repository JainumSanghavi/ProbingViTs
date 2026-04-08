Group Members: Jainum Sanghavi
Title: Probing Vision Transformers to Extract Structural and Spatial Information
Problem Statement
Vision Transformers perform well on classification, detection, and segmentation, all of which implicitly require understanding of spatial structure: where objects are, how far surfaces are, where regions begin and end. Yet none of this is directly supervised. A ViT trained on ImageNet only sees a class label.
We want to know whether spatial properties (object position, depth) and structural properties (object boundaries) are linearly encoded in a ViT's intermediate representations, and at which layers they emerge.
Success looks like a clear layerwise picture of when and where these properties become linearly recoverable, and a comparison between spatial and structural signals to see if they arise at different depths.
Proposed Approach
We extract patch-level hidden states from each transformer block of a pretrained ViT and train a linear probe at each layer to predict a patch-level label.
The primary experiment would use BSDS500 to probe for object boundaries, assigning each patch a label indicating whether it overlaps a ground truth boundary contour. This tests local structural sensitivity that is entirely absent from the training signal.
Boundary may not necessarily be categorized as a spatial feature. I may consider my experiment by replacing boundary prediction with depth prediction. As of now, depth is an extension if time permits. The dataset that can be used for learning/decoding depth is NYU Depth V2 with discretized depth bins.
If time permits, we add activation patching at the peak encoding layer to test whether boundary information is causally used by the model.
Background Work:
Alain and Bengio (2017) established the theoretical basis for linear probing on frozen representations. Hewitt and Manning (2019) extended this with structural probes, showing that entire syntax trees are implicitly embedded in BERT's vector geometry, which informs our approach of probing for structured spatial properties. Hewitt and Liang (2019) raise the key limitation: probes recover information the model may not actually use, making results correlational. Lipton (2018) broadly cautions against underspecified interpretability claims. Heimersheim and Nanda (2024) provide practical guidance on activation patching as a causal follow-up to probing, which we draw on if time permits.
On the ViT side, Dosovitskiy et al. (2020) introduced the architecture, Caron et al. (2021) showed DINO-trained ViTs develop object-segmenting attention without segmentation supervision, and Raghu et al. (2021) found ViTs develop global structure earlier than CNNs. More recently, Chowdhury et al. (2025) showed spatially localized information is recoverable from frozen patch embeddings via lightweight probing, Zhou et al. (2024) showed same-object patches cluster together in representation space across layers, and Qiang et al. (2025) showed interpretability can be explicitly trained into a ViT rather than recovered post-hoc.
Thi project applies structured probing with ground truth boundary labels to ask whether specific structural properties are linearly encoded, and whether that encoding is causally relevant.
Evaluation Plan:
Datasets: BSDS500 (primary), MS-COCO and NYU Depth V2 subsets (if time permits).
Model: ViT-B/16 pretrained on ImageNet-21k via HuggingFace Transformers.
Metrics: Accuracy and F1 at each layer. Precision-recall for boundaries given class imbalance.
Baselines: Random ViT for the floor, majority class classifier as a trivial ceiling, nonlinear MLP probe as an upper bound on recoverable information.
Sanity check: Probe accuracy on the first layer (before any attention) should be near baseline. High accuracy there signals a label construction error.
What success looks like: Probe accuracy significantly exceeds a majority-class baseline at one or more layers, with a clear layer-wise trend.
What failure looks like: Uniformly low accuracy across all layers means either the property is not linearly encoded, or label construction is noisy. If probes fail to beat baseline, that suggests ViTs achieve strong performance without linearly encoding these properties, raising questions about what they do encode. 
Risk & Mitigation: 
Linear probes may be too weak to decode spatially distributed boundary information, since a single scalar decision boundary may not capture the structure of patch representations. If probe accuracy is consistently low across all layers, we switch to a lightweight convolutional probe that operates over local neighborhoods of patches.
Probing results are inherently correlational. To address this, we check whether probe accuracy covaries with model performance across layers as a consistency test, and if time permits, run activation patching to test whether boundary-encoding layers are causally relevant to model outputs.
Compute requirements may exceed what my M3 Mac can handle comfortably for larger experiments. In that case, we move to Google Colab or the Discovery cluster as fallback options.
BSDS500 has known label noise since different annotators mark boundaries inconsistently. To mitigate this, we aggregate annotations across all annotators before assigning patch labels, treating a patch as a boundary patch only if a majority of annotators agree.
Timeline:
Week 1: Download BSDS500, build patch-level label pipeline, verify alignment visually.
Week 2: Extract and cache ViT hidden states for all layers.
Week 3: Train linear probes layerwise, run random ViT baseline, generate accuracy curves.
Week 4: Run CLS vs patch token ablation. Begin MS-COCO and NYU experiments if on schedule.
Week 5: Analyze results, produce visualizations, write report, prepare presentation.


