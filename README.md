### Dataset overview
The dataset represents diagnostic measurements derived from digitized images of the fine needle aspirate (FNA) samples of breast masses. The features capture various geometric and stadistical properties of cell present in the images, **categorized into three main calculations**:
* **(1) Mean**: The average measurement for each feature across the sample 
    * sufix: `_mean`
* **(2) Standard Error (se)**: The variability or error in the measurements for each feature
    * sufix: `_se`
* **(3) Worst**: The largest measurement, typically computed as the mean of the three largest values 
    * sufix: `_worst`

### Columns *(base attributes)*
* **(1) ID**: identifier number for each patient/sample *(not relevant for the predictive analysis)*
* **(2) diagnosis**: Target value. Indicates wether the sumor is **M** $\rightarrow$ malignant (cancerous), **B** $\rightarrow$ benign (non cancerous). 
    * From now on, this values will be considerate as their mapped values:
        * **B** $= 0$
        * **M** $= 1$
* **(3) radius**: mean distance from the center of the nucleous to its perimeter
    * columns: `radius_mean`, `radius_se`, `radius_worst`
* **(4) texture**: measures the variation un te grayscale pixel intensity within the cell nucleus
    * columns: `texture_mean`,`texture_se`,`texture_worst`
* **(5) permimeter**: length of the boundary *(límite)* of the cell nucleous
    * columns: `perimeter_mean`,`perimeter_se`,`perimeter_worst`
* **(6) area**: total area enclosed within the cell nucleous boundary
    * column: `area_mean`,`area_se`,`area_worst`
* **(7) smoothness**: quantifies the local variation in radius length across the cell nucelous boundary
    * column: `smoothness_mean`,`smoothness_se`,`smoothness_worst`
* **(8) compactness**: represents the compactness of the nucleous, computed as: $(\frac{\text{perimeter}^2}{\text{area}-1})$
    * columns: `compactness_mean`,`compactness_se`,`compactness_worst`
* **(9) concavity**: Measures the severity of the concave portions on the contour of the nucleous
    * columns: `concavity_mean`,`concavity_se`,`concavity_worst`
* **(10) concave points**: counts the number of the concave points on the nucleous boundary
    * columns: `concave_points_mean`,`concave_points_se`,`concave_points_worst`
* **(11) symmetry**: quantifies the symmetry of the nucleus shape
    * columns: `symmetry_mean`,`symmetry_se`,`symmetry_worst`
* **(12) fractal dimension**: measures the complexity of the nucleus boundary, computed using a *coastline approximation*
    * columns: `fractal_dimension_mean`,`fractal_dimension_se`,`fractal_dimension_wrost`
* **Unnamed 32**: Likely empty/irrelevant column for this analysis

### target class distribution
**B**: $357$ samples,  **M**: $212$ samples
* since the number of the samples is not the same for both cases *(benign vs malignant)* the possibility of balancing will be analyced.


### Importance in medical context:
In medical applications *(especially on tumor detection)* it's critical to minimize false negatives. A $95$% recall for the malignant class means that the model fails to detect only $5$% of malignant cases, which is highly desirable.

**Overall (global) vs class comparison**: While overall recall $0.947$ is important, what we're really interested is the specific recall for the "malignant" class.
* If that metric reaches $95$% or higher, our goal is met *(even if the overall is slightly lower)*

**Strategy applied during the training**: Given the high cost of false negatives (FN) in medicine, focusing on maximizing the recall of the "malignant" class is a priority. This has been achieved by tuning class weights and using a custom scorer, which is a good practice in this context.

