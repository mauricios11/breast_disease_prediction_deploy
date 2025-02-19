#
# current path: .utils/utils_categorical_plots.py

#libraries
import math
import matplotlib.pyplot as plt
import numpy   as np
import pandas  as pd
import seaborn as sns
from typing import List, Tuple, Dict, Union

class CategoricalPlots:
    def __init__(self) -> None:
        pass
    
    #-Func 01 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#g       


    def violin_strip(self, df: pd.DataFrame,
                     x: str,
                     y: str, hue: str = None,
                     split: bool = True,
                     palette_strip: str | dict = None,
                     palette_violin: str | dict = 'light:white',
                     fig_size: Tuple = (12, 6),
                     bw_method: str = 0.6): # loc_params: str = None, label_on: bool = False):
        """ Plots a swarmplot & stripplot (categorical vs numerical)
            - A bivariate analysis between two columns of a DataFrame
            - the columns MUST be one categorical and one numerical (no matter the order)
        
            Args: 
            - df (pd.DataFrame)-----: The DataFrame to plot.
            - x (str)---------------: The column to plot on the x-axis.
            - y (str)---------------: The column to plot on the y-axis.
            - hue (str, optional)---: The column to use for the color encoding.
            - split (bool, optional): If True, split the violin plot. Defaults to -> True.
            - palette_strip ()------: The color palette to use in the swarmplot.
            - palette_violin ()-----: The color palette to use in the violinplot.
            
        raises:
            - TypeError: If the input is not a DataFrame or at least one of the columns is not numeric.
            - ValueError: If x or y are not in the DataFrame.
            - TypeError: If x and y are not one numeric and one categorical or object.
        
        example: stat_plot.violin_strip(
                        df= df_cars, x= 'price_usd', y= 'body_type', hue= 'engine_type', split= True,
                        palette_strip= 'viridis', palette_violin= 'viridis', bw_method= 0.6)"""
                        
        # Validaciones
        if not isinstance(df, pd.DataFrame):
            raise TypeError('ERROR: df must be a DataFrame')
        if x not in df.columns or y not in df.columns:
            raise ValueError('ERROR: x or y not in DataFrame')

        # Verificar tipos de datos
        is_x_numeric = pd.api.types.is_numeric_dtype(df[x])
        is_y_numeric = pd.api.types.is_numeric_dtype(df[y])
        is_x_categorical = isinstance(df[x], pd.CategoricalDtype) or pd.api.types.is_object_dtype(df[x])
        is_y_categorical = isinstance(df[x], pd.CategoricalDtype) or pd.api.types.is_object_dtype(df[y])

        if not (is_x_numeric or is_y_numeric):
            raise TypeError('ERROR: At least one of x or y must be numeric')
        if not (is_x_categorical or is_y_categorical):
            raise TypeError('ERROR: At least one of x or y must be categorical or object')

        # verificar que haya una categórica y una numérica en los ejes
        x_cat_vs_y_num = is_x_categorical and is_y_numeric
        x_num_vs_y_cat = is_x_numeric and is_y_categorical
        
        plt.figure(figsize= fig_size)
         
        if x_cat_vs_y_num or x_num_vs_y_cat:
            sns.violinplot(data= df, x= x, y= y, hue= hue,
                           split=split,
                           inner='quartile',
                           bw_method= bw_method, # precisión del KDE
                           palette= palette_violin, zorder= 1, alpha= 0.4)
            
            sns.stripplot(data= df, x= x, y= y, hue= hue,
                          jitter= True,
                          dodge= True, size= 6, linewidth= 0.1, marker= '^',
                          palette= palette_strip, zorder= 2, alpha= 0.4)

        else:
            raise TypeError('ERROR: x and y must be one numeric and one categorical or object')

        plt.title(f'{y} by {x}', fontsize= 16, weight= 'bold')
        plt.tick_params(labelsize= 13)
        plt.xlabel(x, fontsize= 14, weight= 'bold')
        plt.ylabel(y, fontsize= 14, weight= 'bold')

        plt.tight_layout()
        plt.show()
        
        
    #-Func 02 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#comprobar categóricos
    def plot_dist(self, df:pd.DataFrame,
                  cols: List[str],
                  plot_selection: str= 'histplot',
                  hist_params: Dict[str,Union[str,bool,int]]= None,
                  stats: bool= False,
                  palette: Union[str, Dict]= None,
                  multiple: str= 'stack',
                  title: str= None):
        """Plot distributions or boxplots for a list of columns with additional options.
        
        Args:
        - df (pd.DataFrame).........: DataFrame to analyze.
        - cols (List[str])..........: Columns to plot.
        - plot_selection (str)......: Type of plot ('histplot' or 'boxplot').
        - hist_params (Dict)........: Parameters for histplot (keys: 'hue', 'kde', 'bins').
        - stats (bool)..............: If True, plot mean, median, Q1, Q3 as vertical lines.
        - palette (Union[str, Dict]): Color palette for the plots.
        - multiple (str)............: Multiple parameter for histplot (e.g., 'stack', 'dodge').
        - title (str)...............: Title for the entire plot.
        - fig_size (Tuple[int, int]): Size of the figure.
        """
        # validations
        if plot_selection not in ['histplot', 'boxplot']:
            raise ValueError('ERROR: plot_selection must be "histplot" or "boxplot"')
        
        # preset params (for histplot)
        if not hist_params:
            hist_params = {'hue': None, 'kde': True, 'bins': 50, 'palette': 'viridis'}
        
        # filter numeric only
        numeric_cols = [col for col in cols if np.issubdtype(df[col].dtype, np.number)]
        if not numeric_cols:
            raise ValueError('ERROR: At least one column must be numeric')
        
        # subplot config
        num_cols = len(numeric_cols)
        cols_num = min(3, num_cols) # max 3 columns per row
        rows_num = math.ceil(num_cols / cols_num)
        fig, axes = plt.subplots(rows_num, cols_num, figsize= (cols_num * 5, rows_num * 4))
        axes = np.array(axes).flatten() # make sure 'axes' is a 1D array
        
        # plots
        for i, col in enumerate(numeric_cols):
                ax = axes[i]
                if plot_selection == 'histplot':
                    sns.histplot(
                        data= df,
                        x= col, 
                        hue=  hist_params.get('hue'),
                        kde=  hist_params.get('kde'), 
                        bins= hist_params.get('bins'),
                        palette=palette,
                        multiple=multiple,
                        ax=ax, alpha=0.4
                    )
                    ax.set_title(col, fontsize=16)
                    ax.set_xlabel(col, fontsize=14)
                    ax.set_ylabel('Count', fontsize=14)
                    
                    if stats:
                        mean_val = df[col].mean()
                        median_val = df[col].median()
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        ax.axvline(mean_val, color='orange', linestyle='-.', linewidth=1.5, label='Mean')
                        ax.axvline(median_val, color='purple', linestyle='-.', linewidth=1.5, label='Median')
                        ax.axvline(q1, color='blue', linestyle='-', linewidth=1.2)
                        ax.axvline(q3, color='blue', linestyle='-', linewidth=1.2)
                        ax.legend()

                elif plot_selection == 'boxplot':
                    sns.boxplot(data=df, x=col, palette=palette, ax=ax)
                    ax.set_title(col, fontsize=16)
                    ax.set_xlabel(col, fontsize=14)
            
        # Ajustar diseño
        if title:
            fig.suptitle(title.upper(), fontsize=20, weight='bold')
        plt.tight_layout()
        plt.show()      
    
    #-Func 03 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#       
    """Plots a countplot for each column in the list
        - The count of ocurrences for each category in the column
        Args:
            - df ----------------------: The DataFrame to plot.
            - cols --------------------: The columns to plot.
            - rotation (int, optional)-: The rotation of the x-axis labels. Defaults to None.
            - bar_nums (bool, optional): If True, show the value above each bar. Defaults to True.
        
        Raises: ValueError: If the input df is not a df.
        Example:
            plot_cat_feq(df= df_cars, cols= ['',''], rotation= 0, bar_nums= True)"""
    def plot_cat_feq(self, df: pd.DataFrame,
                     cols: list,
                     rotation: int= None,
                     bar_nums: bool= True,
                     palette= 'viridis'):
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df must be a pandas DataFrame")
        
        def annotate_bars(ax): #------------------------------------------- auxiliar func: value above each bar
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha= 'center', va= 'center', fontsize= 8,
                            color='black', xytext= (0, 5),
                            textcoords= 'offset points')

        num_cols = len(cols)
        cols_num = min(3, num_cols) #-------------------------------------- establish the size for the plot grid (subplot)
        rows_num  = math.ceil(num_cols / cols_num) 
        
        fig, axes = plt.subplots(rows_num, cols_num,
                                 figsize= (7 * cols_num, 6 * rows_num))
        axes = np.array(axes).flatten() #---------------------------------- make sure "axes" is a 1d array
        for i, col in enumerate(cols):
            ax = axes[i]
            sns.countplot(data= df, x= col, ax= ax,
                          hue= col, palette= palette)
            ax.set_title(f'Count of ocurrences for: {col}',
                        fontweight= 'bold', fontsize= 15, loc= 'left')
            
            if rotation:
                ax.tick_params(axis= 'x', rotation= rotation)
            if bar_nums:
                annotate_bars(ax)
                    
        for j in range(i +1, len(axes)):
            fig.delaxes(axes[j]) # --------------------------------------- remove empty plots
            
        plt.tight_layout() # --------------------------------------------- adjust the layout: avoid overlap
        plt.show()
        
    #-Func 04 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 
    def multiple_boxplots(self, df,
                          cols: List[str],
                          target: str,
                          plot_per_row:int= 3,
                          rotation: int= 90):
        """plot multiple boxplots. works better when target is numerical"""
        number_cols = len(cols)
        cols_num = min(plot_per_row, number_cols) #-------------------------------------- establish the size for the plot grid (subplot)
        rows_num  = math.ceil(number_cols / cols_num) 
                
        fig, axes = plt.subplots(rows_num, cols_num,
                                    figsize= (7 * cols_num, 6 * rows_num))
            
        axes = np.array(axes).flatten() #---------------------------------- make sure "axes" is a 1d array

        for i, col in enumerate(cols):
            ax = axes[i]
            sns.boxplot(x=col, y= target, data=df, color='gold', ax=ax)
            ax.set_title(f'{col.upper()} vs {target} distribution', fontsize= 12,
                         fontweight= 'bold', loc= 'left', color= 'gray')
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    #-Func 05 #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
    def multiple_histplots(self, df,
                          cols: List[str],
                          target: str,
                          plot_per_row:int= 3,
                          rotation: int= 90,
                          use_y: bool= True,
                          fig_size: List[int]= [7, 6],
                          stats: bool= True):
        """ make docstring
        """
        number_cols = len(cols)
        cols_num = min(plot_per_row, number_cols) #-------------------------------------- establish the size for the plot grid (subplot)
        rows_num  = math.ceil(number_cols / cols_num) 
                
        fig, axes = plt.subplots(rows_num, cols_num,
                                 figsize= (fig_size[0] * cols_num, fig_size[1] * rows_num))
            
        axes = np.array(axes).flatten() #---------------------------------- make sure "axes" is a 1d array

        for i, col in enumerate(cols):
            ax = axes[i]
            if use_y:
                sns.histplot(x=col, y= target, data=df, color='gold', ax=ax)
            else: 
                sns.histplot(x=col, data= df, hue= target, ax= ax, kde= True)
            
            if stats:
                        mean_val = df[col].mean()
                        median_val = df[col].median()
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        ax.axvline(mean_val, color='orange', linestyle='-.', linewidth=1.5, label='Mean')
                        ax.axvline(median_val, color='purple', linestyle='-.', linewidth=1.5, label='Median')
                        ax.axvline(q1, color='blue', linestyle='-', linewidth=1.2)
                        ax.axvline(q3, color='blue', linestyle='-', linewidth=1.2)
                        ax.legend()
               
            ax.set_title(f'{col.upper()} vs {target} distribution', fontsize= 12,
                          fontweight= 'bold', loc= 'left', color= 'gray')
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()



