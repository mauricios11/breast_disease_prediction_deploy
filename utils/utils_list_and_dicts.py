# module made for storing lists and dictionaries used in the project
# current path: .utils/list_and_dicts.py

# libraries

# sorted by base atributes
class ListAndDicts:
    def __init__(self) -> None:
        
        # sorted columns by attribute (all columns)
        self.sorted_all = [
                  'id', 'Unnamed: 32',                                      # lol columns :D
                  'diagnosis',                                              # target
                  'radius_mean', 'radius_se', 'radius_worst',               # attribute: radius
                  'texture_mean', 'texture_se', 'texture_worst',            # attribute: texture
                  'perimeter_mean', 'perimeter_se', 'perimeter_worst',      # attribute: perimeter
                  'area_mean', 'area_se', 'area_worst',                     # attribute: area
                  'smoothness_mean', 'smoothness_se', 'smoothness_worst',   # attribute: smoothness
                  'compactness_mean', 'compactness_se', 'compactness_worst',# attribute: compactness
                  'concavity_mean', 'concavity_se', 'concavity_worst',      # attribute: concavity    
                  'concave points_mean', 'concave points_se',               # attribute: concave points
                  'concave points_worst',                                   # attribute: concave points
                  'symmetry_mean', 'symmetry_se', 'symmetry_worst',         # attribute: symmetry
                  'fractal_dimension_mean', 'fractal_dimension_se',         # attribute: fractal dimension
                  'fractal_dimension_worst'                                 # attribute: fractal dimension
                  ]
        
        # sorted columns (order used in training)
        self.column_names_in_training = [
            'radius_mean', 'texture_mean', 'perimeter_mean',                      # mean values
            'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', # mean values
            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',     # mean values
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',# standard error values
            'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', # standard error values
            'fractal_dimension_se',                                               # standard error values
            'radius_worst', 'texture_worst',                                      # worst values
            'perimeter_worst', 'area_worst', 'smoothness_worst',                  # worst values
            'compactness_worst', 'concavity_worst', 'concave points_worst',       # worst values
            'symmetry_worst', 'fractal_dimension_worst'                           # worst values
        ]
        # new data for prediction
        self.new_data_for_prediction = {

            'area_se'    : 153.4,
            'radius_se'  : 1.095,
            'area_mean'  : 1001,
            'texture_se' : 0.9053,
            'area_worst' : 2019,
            'radius_mean': 17.99,
            'symmetry_se': 0.03003,
            'texture_mean'  : 10.38,
            'perimeter_se'  : 8.589,
            'concavity_se'  : 0.0537,
            'radius_worst'  : 25.38,
            'texture_worst' : 17.33,
            'symmetry_mean' : 0.2419,
            'smoothness_se' : 0.006399,
            'concavity_mean': 0.3001,
            'perimeter_mean': 122.8,
            'compactness_se': 0.04904,
            'symmetry_worst'   : 0.4601,
            'perimeter_worst'  : 184.6,
            'concavity_worst'  : 0.7119,
            'smoothness_mean'  : 0.1184,
            'smoothness_worst' : 0.1622,
            'compactness_mean' : 0.2776,
            'concave points_se': 0.01587,
            'compactness_worst': 0.6656,
            'concave points_mean'    : 0.1471,
            'fractal_dimension_se'   : 0.006193,
            'concave points_worst'   : 0.2654,
            'fractal_dimension_mean' : 0.07871,
            'fractal_dimension_worst': 0.1189
        }
        
        
        
        
        
        
        