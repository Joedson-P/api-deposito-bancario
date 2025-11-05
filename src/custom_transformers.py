from sklearn.base import BaseEstimator, TransformerMixin


class BinaryMapper(BaseEstimator, TransformerMixin):
    """
    Transformador customizado para mapear colunas binárias ('yes'/'no') para numérico (1/0).
    Herda de BaseEstimator e TransformerMixin para integração com Pipelines do Scikit-learn.
    """
    def __init__(self, categories={'yes': 1, 'no': 0}):
        self.categories = categories

    def fit(self, X, y=None):
        """
        O BinaryMapper não precisa aprender parâmetros dos dados.
        Retorna 'self' para manter a interface do Scikit-learn.
        """
        return self

    def transform(self, X):
        """
        Aplica o mapeamento 'yes' -> 1 e 'no' -> 0 às colunas de entrada.
        
        Args:
            X (pd.DataFrame): DataFrame contendo as colunas binárias.

        Returns:
            pd.DataFrame: DataFrame com os valores mapeados para 1 e 0.
        """
        return X.apply(lambda col: col.map(self.categories))
    
    def get_feature_names_out(self, names=None):
        """
        Retorna os nomes das features de saída.
        Como o mapeamento binário não altera o nome da coluna, 
        retorna os nomes originais.
        """
        return names

class MonthMapper(BaseEstimator, TransformerMixin):
    """
    Transformador customizado para mapear nomes de meses para valores ordinais (1 a 12).
    """
    def __init__(self):
        self.month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        # Mapeamento de str para int (1 a 12)
        self.month_map = {month: i + 1 for i, month in enumerate(self.month_order)}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Aplica o mapeamento ordinal (mês -> número) à coluna 'month'.
        
        Args:
            X (pd.DataFrame): DataFrame contendo a coluna 'month'.

        Returns:
            pd.DataFrame: DataFrame com a coluna 'month' mapeada para valores ordinais.
        """
        # X.iloc[:, 0] é usado porque o ColumnTransformer passa apenas a coluna selecionada
        return X.iloc[:, 0].map(self.month_map).to_frame() # Mapeia e retorna como DataFrame
    
    def get_feature_names_out(self, names=None):
        """
        Retorna os nomes das features de saída.
        A transformação ordinal de mês não altera o nome da coluna de entrada.
        """
        return names