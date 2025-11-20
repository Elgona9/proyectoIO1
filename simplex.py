"""
Implementación de métodos de optimización lineal:
- Método Simplex
- Método de la Gran M (Big M)
- Método de las Dos Fases (Two Phase)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class SimplexSolver:
    """Clase base para resolver problemas de programación lineal con el método Simplex"""
    
    def __init__(self, c: List[float], A: List[List[float]], b: List[float], 
                 constraints: List[str], optimize: str = 'max'):
        """
        Inicializa el problema de programación lineal
        
        Args:
            c: Coeficientes de la función objetivo
            A: Matriz de coeficientes de las restricciones
            b: Términos independientes de las restricciones
            constraints: Lista de tipos de restricciones ('<=', '>=', '=')
            optimize: Tipo de optimización ('max' o 'min')
        """
        self.c_original = np.array(c, dtype=float)
        self.A_original = np.array(A, dtype=float)
        self.b_original = np.array(b, dtype=float)
        self.constraints = constraints
        self.optimize = optimize.lower()
        
        # Variables para almacenar el resultado
        self.solution = None
        self.optimal_value = None
        self.iterations = []
        self.status = None
        
    def solve(self) -> Dict:
        """Método abstracto que debe ser implementado por las subclases"""
        raise NotImplementedError("Este método debe ser implementado por las subclases")
    
    def _is_optimal(self, tableau: np.ndarray) -> bool:
        """Verifica si la tabla es óptima"""
        # Todos los coeficientes en la fila objetivo deben ser >= 0 (minimización canónica)
        return np.all(tableau[-1, :-1] >= -1e-10)
    
    def _find_pivot_column(self, tableau: np.ndarray) -> Optional[int]:
        """Encuentra la columna pivote"""
        # Seleccionar la columna con el valor más negativo en la fila objetivo
        obj_row = tableau[-1, :-1]
        if np.all(obj_row >= -1e-10):
            return None
        return np.argmin(obj_row)
    
    def _find_pivot_row(self, tableau: np.ndarray, pivot_col: int) -> Optional[int]:
        """Encuentra la fila pivote usando la regla del mínimo cociente"""
        column = tableau[:-1, pivot_col]
        rhs = tableau[:-1, -1]
        
        # Calcular ratios solo para elementos positivos
        ratios = []
        for i, val in enumerate(column):
            if val > 1e-10:
                ratios.append((rhs[i] / val, i))
            else:
                ratios.append((float('inf'), i))
        
        if not ratios or min(ratios)[0] == float('inf'):
            return None
        
        return min(ratios)[1]
    
    def _pivot(self, tableau: np.ndarray, pivot_row: int, pivot_col: int) -> np.ndarray:
        """Realiza la operación de pivote"""
        tableau = tableau.copy()
        
        # Dividir la fila pivote por el elemento pivote
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row] = tableau[pivot_row] / pivot_element
        
        # Hacer ceros en la columna pivote
        for i in range(len(tableau)):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                tableau[i] = tableau[i] - factor * tableau[pivot_row]
        
        return tableau
    
    def _extract_solution(self, tableau: np.ndarray, num_original_vars: int) -> Tuple[np.ndarray, float]:
        """Extrae la solución de la tabla final"""
        solution = np.zeros(num_original_vars)
        
        for j in range(num_original_vars):
            col = tableau[:-1, j]
            # Verificar si es una columna básica (una columna con un 1 y el resto 0s)
            if np.count_nonzero(np.abs(col - 1) < 1e-10) == 1 and np.count_nonzero(np.abs(col) < 1e-10) == len(col) - 1:
                idx = np.argmax(np.abs(col - 1) < 1e-10)
                solution[j] = tableau[idx, -1]
        
        # El valor en tableau[-1, -1] depende de cómo configuramos el problema:
        # - Para MAX: convertimos a MIN negando c, entonces tableau[-1,-1] ya es el valor correcto (positivo)
        # - Para MIN: el valor está negado en la tabla estándar, así que lo negamos
        if self.optimize == 'max':
            optimal_value = tableau[-1, -1]
        else:
            optimal_value = -tableau[-1, -1]
        
        return solution, optimal_value
    
    def _extract_all_variables(self, tableau: np.ndarray) -> Dict:
        """
        Extrae los valores de todas las variables (decisión, holgura y artificiales)
        
        Returns:
            Dict con variables de decisión, holgura y artificiales
        """
        n_vars = tableau.shape[1] - 1  # Total de columnas menos RHS
        all_values = np.zeros(n_vars)
        
        for j in range(n_vars):
            col = tableau[:-1, j]
            # Verificar si es una columna básica
            if np.count_nonzero(np.abs(col - 1) < 1e-10) == 1 and np.count_nonzero(np.abs(col) < 1e-10) == len(col) - 1:
                idx = np.argmax(np.abs(col - 1) < 1e-10)
                all_values[j] = tableau[idx, -1]
        
        return all_values

    def format_tableau(self, tableau: np.ndarray, cj: Optional[List[float]] = None,
                       var_names: Optional[List[str]] = None, precision: int = 4) -> str:
        """
        Formatea una tabla simplex para impresión humana con las filas:
        - Cabecera CJ (valores de las variables)
        - Columnas: VB (variable básica), CB (coeficiente de la variable básica), luego las variables y BJ (RHS)
        - Filas para cada variable básica
        - Filas ZJ y CJ-ZJ

        Args:
            tableau: matriz numpy de la tabla (última columna RHS, última fila función objetivo)
            cj: lista de coeficientes CJ (longitud = num_cols - 1). Si None, se inferirán de la fila objetivo.
            var_names: nombres de columnas variables (longitud = num_cols - 1). Si None, se generan X1.., S1.., ...
            precision: número de decimales para representar valores.

        Returns:
            Una cadena con la tabla formateada.
        """
        rows, cols = tableau.shape
        n_rows = rows - 1
        n_vars = cols - 1

        # Inferir CJ si no se proporciona
        if cj is None:
            cj_arr = list(tableau[-1, :n_vars].astype(float))
        else:
            cj_arr = [float(x) for x in cj]
            # if length mismatch, try to pad/truncate
            if len(cj_arr) < n_vars:
                cj_arr = cj_arr + [0.0] * (n_vars - len(cj_arr))
            elif len(cj_arr) > n_vars:
                cj_arr = cj_arr[:n_vars]

        # Generate variable names if needed
        if var_names is None:
            orig = len(self.c_original)
            names = []
            for i in range(n_vars):
                if i < orig:
                    names.append(f"x{i+1}")
                else:
                    names.append(f"s{i - orig + 1}")
        else:
            names = var_names[:n_vars]
            if len(names) < n_vars:
                names = names + [f"v{i+1}" for i in range(len(names), n_vars)]

        # Detect basic columns: map row -> column index if column is unit vector
        basic_col_for_row = {i: None for i in range(n_rows)}
        for j in range(n_vars):
            col = tableau[:-1, j]
            # check if column is unit vector
            is_one = np.isclose(col, 1.0, atol=1e-8)
            is_zero = np.isclose(col, 0.0, atol=1e-8)
            if np.count_nonzero(is_one) == 1 and np.count_nonzero(~is_zero) == 1:
                row_idx = int(np.where(is_one)[0][0])
                basic_col_for_row[row_idx] = j

        # Compute CB per row
        cb_list = []
        vb_list = []
        for i in range(n_rows):
            col_idx = basic_col_for_row.get(i)
            if col_idx is None:
                cb_list.append(0.0)
                vb_list.append("-")
            else:
                cb_list.append(cj_arr[col_idx])
                vb_list.append(names[col_idx])

        # Compute Zj
        zj = [0.0] * n_vars
        for j in range(n_vars):
            s = 0.0
            for i in range(n_rows):
                s += cb_list[i] * float(tableau[i, j])
            zj[j] = s

        # CJ - ZJ
        cj_minus_zj = [cj_arr[j] - zj[j] for j in range(n_vars)]

        # Prepare column widths
        col_texts = ["VB", "CB"] + names + ["BJ"]
        # compute string values matrix for rows
        def fmt(x):
            if abs(x) < 10**(-precision):
                x = 0.0
            # Formatear sin ceros innecesarios
            formatted = f"{x:.{precision}f}"
            # Remover ceros trailing y punto decimal si es entero
            if '.' in formatted:
                formatted = formatted.rstrip('0').rstrip('.')
            return formatted if formatted else '0'

        table_rows = []
        for i in range(n_rows):
            row = []
            row.append(vb_list[i])
            row.append(fmt(cb_list[i]))
            for j in range(n_vars):
                row.append(fmt(float(tableau[i, j])))
            row.append(fmt(float(tableau[i, -1])))
            table_rows.append(row)

        # ZJ and CJ-ZJ rows
        zj_row = ["ZJ", ""] + [fmt(v) for v in zj] + [fmt( sum(cb_list[i] * float(tableau[i, -1]) for i in range(n_rows)) )]
        cjzj_row = ["CJ-ZJ", ""] + [fmt(v) for v in cj_minus_zj] + [fmt(0.0)]

        # Header CJ line
        header_cj = ["CJ", ""] + [fmt(v) for v in cj_arr] + [""]

        # Column widths
        cols_count = 2 + n_vars + 1
        widths = [max(len(col_texts[k]),  max((len(r[k]) for r in table_rows), default=0), len(str(header_cj[k])) if k < len(header_cj) else 0)
                  for k in range(cols_count)]

        # ensure CB width at least 4
        widths[1] = max(widths[1], 4)

        # Build formatted lines
        lines = []
        # CJ header
        line = f"{header_cj[0]:>{widths[0]}} {header_cj[1]:>{widths[1]}}"
        for j in range(n_vars):
            line += " " + f"{header_cj[2 + j]:>{widths[2 + j]}}"
        line += " " + f"{header_cj[-1]:>{widths[-1]}}"
        lines.append(line)

        # Second header with VB CB and variable names
        line = f"{col_texts[0]:>{widths[0]}} {col_texts[1]:>{widths[1]}}"
        for j in range(n_vars):
            line += " " + f"{col_texts[2 + j]:>{widths[2 + j]}}"
        line += " " + f"{col_texts[-1]:>{widths[-1]}}"
        lines.append(line)

        # Divider
        lines.append("-" * (sum(widths) + (cols_count - 1)))

        # Body rows
        for r in table_rows:
            line = f"{r[0]:>{widths[0]}} {r[1]:>{widths[1]}}"
            for j in range(n_vars):
                line += " " + f"{r[2 + j]:>{widths[2 + j]}}"
            line += " " + f"{r[-1]:>{widths[-1]}}"
            lines.append(line)

        # Divider
        lines.append("-" * (sum(widths) + (cols_count - 1)))

        # ZJ and CJ-ZJ
        line = f"{zj_row[0]:>{widths[0]}} {zj_row[1]:>{widths[1]}}"
        for j in range(n_vars):
            line += " " + f"{zj_row[2 + j]:>{widths[2 + j]}}"
        line += " " + f"{zj_row[-1]:>{widths[-1]}}"
        lines.append(line)

        line = f"{cjzj_row[0]:>{widths[0]}} {cjzj_row[1]:>{widths[1]}}"
        for j in range(n_vars):
            line += " " + f"{cjzj_row[2 + j]:>{widths[2 + j]}}"
        line += " " + f"{cjzj_row[-1]:>{widths[-1]}}"
        lines.append(line)

        return "\n".join(lines)
    
    def format_tableau_html(self, tableau: np.ndarray, cj: Optional[List[float]] = None,
                           var_names: Optional[List[str]] = None, precision: int = 4,
                           pivot_row: Optional[int] = None, pivot_col: Optional[int] = None,
                           entering_row: Optional[int] = None) -> str:
        """
        Formatea una tabla simplex como HTML con estilos inline.
        
        Args:
            tableau: matriz numpy de la tabla
            cj: lista de coeficientes CJ
            var_names: nombres de columnas variables
            precision: número de decimales
            pivot_row: fila del pivote (para resaltar)
            pivot_col: columna del pivote (para resaltar)
            entering_row: fila que entra en la base (para resaltar diferente)
            
        Returns:
            String con HTML de la tabla formateada
        """
        rows, cols = tableau.shape
        n_rows = rows - 1
        n_vars = cols - 1

        # Inferir CJ si no se proporciona
        if cj is None:
            cj_arr = list(tableau[-1, :n_vars].astype(float))
        else:
            cj_arr = [float(x) for x in cj]
            if len(cj_arr) < n_vars:
                cj_arr = cj_arr + [0.0] * (n_vars - len(cj_arr))
            elif len(cj_arr) > n_vars:
                cj_arr = cj_arr[:n_vars]

        # Generate variable names if needed
        if var_names is None:
            orig = len(self.c_original)
            names = []
            for i in range(n_vars):
                if i < orig:
                    names.append(f"X<sub>{i+1}</sub>")
                else:
                    names.append(f"S<sub>{i - orig + 1}</sub>")
        else:
            names = var_names[:n_vars]

        # Detect basic columns
        basic_col_for_row = {i: None for i in range(n_rows)}
        for j in range(n_vars):
            col = tableau[:-1, j]
            is_one = np.isclose(col, 1.0, atol=1e-8)
            is_zero = np.isclose(col, 0.0, atol=1e-8)
            if np.count_nonzero(is_one) == 1 and np.count_nonzero(~is_zero) == 1:
                row_idx = int(np.where(is_one)[0][0])
                basic_col_for_row[row_idx] = j

        # Compute CB per row
        cb_list = []
        vb_list = []
        for i in range(n_rows):
            col_idx = basic_col_for_row.get(i)
            if col_idx is None:
                cb_list.append(0.0)
                vb_list.append("-")
            else:
                cb_list.append(cj_arr[col_idx])
                vb_list.append(names[col_idx])

        # Compute Zj
        zj = [0.0] * n_vars
        for j in range(n_vars):
            s = 0.0
            for i in range(n_rows):
                s += cb_list[i] * float(tableau[i, j])
            zj[j] = s

        # CJ - ZJ
        cj_minus_zj = [cj_arr[j] - zj[j] for j in range(n_vars)]

        def fmt(x):
            if abs(x) < 10**(-precision):
                x = 0.0
            # Formatear sin ceros innecesarios
            formatted = f"{x:.{precision}f}"
            # Remover ceros trailing y punto decimal si es entero
            if '.' in formatted:
                formatted = formatted.rstrip('0').rstrip('.')
            return formatted if formatted else '0'

        # Build HTML
        html = '<table class="simplex-table">'
        
        # CJ header row
        html += '<thead><tr class="cj-row">'
        html += '<th class="label-cell">C<sub>J</sub></th>'
        html += '<th></th>'
        for v in cj_arr:
            html += f'<th>{fmt(v)}</th>'
        html += '<th></th>'
        html += '</tr>'
        
        # Column headers
        html += '<tr class="header-row">'
        html += '<th class="label-cell">V<sub>B</sub></th>'
        html += '<th class="label-cell">C<sub>B</sub></th>'
        for name in names:
            html += f'<th>{name}</th>'
        html += '<th class="label-cell">B<sub>J</sub></th>'
        html += '</tr></thead>'
        
        # Body rows
        html += '<tbody>'
        for i in range(n_rows):
            row_class = 'data-row'
            if entering_row is not None and i == entering_row:
                row_class = 'data-row entering-row'
            
            html += f'<tr class="{row_class}">'
            html += f'<td class="vb-cell">{vb_list[i]}</td>'
            html += f'<td class="cb-cell">{fmt(cb_list[i])}</td>'
            for j in range(n_vars):
                cell_class = ''
                if pivot_row is not None and pivot_col is not None and i == pivot_row and j == pivot_col:
                    cell_class = ' class="pivot-cell"'
                html += f'<td{cell_class}>{fmt(float(tableau[i, j]))}</td>'
            html += f'<td class="bj-cell">{fmt(float(tableau[i, -1]))}</td>'
            html += '</tr>'
        html += '</tbody>'
        
        # Footer with ZJ and CJ-ZJ
        html += '<tfoot>'
        html += '<tr class="zj-row">'
        html += '<td class="label-cell">Z<sub>J</sub></td>'
        html += '<td></td>'
        for v in zj:
            html += f'<td>{fmt(v)}</td>'
        zj_rhs = sum(cb_list[i] * float(tableau[i, -1]) for i in range(n_rows))
        html += f'<td class="bj-cell">{fmt(zj_rhs)}</td>'
        html += '</tr>'
        
        html += '<tr class="cj-zj-row">'
        html += '<td class="label-cell">C<sub>J</sub>-Z<sub>J</sub></td>'
        html += '<td></td>'
        for v in cj_minus_zj:
            html += f'<td>{fmt(v)}</td>'
        html += f'<td class="bj-cell">{fmt(0.0)}</td>'
        html += '</tr>'
        html += '</tfoot>'
        
        html += '</table>'
        return html

class BigMMethod(SimplexSolver):
    """Método de la Gran M (Big M Method)"""
    
    def solve(self, M: float = 1000.0) -> Dict:
        """
        Resuelve el problema usando el Método de la Gran M
        
        Args:
            M: Valor grande para penalizar variables artificiales
        """
        self.iterations = []
        n_vars = len(self.c_original)
        n_constraints = len(self.b_original)
        
        # Construir la tabla inicial con variables de holgura y artificiales
        tableau, artificial_vars = self._build_initial_tableau(M)
        
        # Guardar iteración inicial
        self.iterations.append({
            'iteration': 0,
            'tableau': tableau.copy(),
            'description': 'Tabla inicial con Método de Gran M'
        })
        
        # Aplicar el método simplex
        iteration = 1
        max_iterations = 100
        
        while not self._is_optimal(tableau) and iteration < max_iterations:
            # Encontrar columna pivote
            pivot_col = self._find_pivot_column(tableau)
            if pivot_col is None:
                self.status = 'optimal'
                break
            
            # Encontrar fila pivote
            pivot_row = self._find_pivot_row(tableau, pivot_col)
            if pivot_row is None:
                self.status = 'unbounded'
                self.iterations.append({
                    'iteration': iteration,
                    'tableau': tableau.copy(),
                    'description': 'Problema no acotado'
                })
                return {
                    'status': 'unbounded',
                    'message': 'El problema no está acotado',
                    'iterations': self.iterations
                }
            
            # Realizar pivote
            tableau = self._pivot(tableau, pivot_row, pivot_col)
            
            self.iterations.append({
                'iteration': iteration,
                'tableau': tableau.copy(),
                'pivot_row': pivot_row,
                'pivot_col': pivot_col,
                'entering_row': pivot_row,  # La fila que entra es la fila del pivote
                'description': f'Pivote en fila {pivot_row + 1}, columna {pivot_col + 1}'
            })
            
            iteration += 1
        
        if iteration >= max_iterations:
            self.status = 'max_iterations'
            return {
                'status': 'error',
                'message': 'Se alcanzó el número máximo de iteraciones',
                'iterations': self.iterations
            }
        
        # Verificar si hay variables artificiales en la base
        for art_var in artificial_vars:
            col = tableau[:-1, art_var]
            if np.any(np.abs(col - 1) < 1e-10):
                # Verificar si el valor es cero
                idx = np.argmax(np.abs(col - 1) < 1e-10)
                if abs(tableau[idx, -1]) > 1e-10:
                    self.status = 'infeasible'
                    return {
                        'status': 'infeasible',
                        'message': 'El problema no tiene solución factible',
                        'iterations': self.iterations
                    }
        
        # Extraer solución
        self.solution, self.optimal_value = self._extract_solution(tableau, n_vars)
        all_vars = self._extract_all_variables(tableau)
        self.status = 'optimal'
        
        # Separar variables por tipo
        n_slack = sum(1 for c in self.constraints if c in ['<=', '>='])
        n_artificial = sum(1 for c in self.constraints if c in ['>=', '='])
        
        slack_vars = all_vars[n_vars:n_vars + n_slack].tolist() if n_slack > 0 else []
        artificial_vars = all_vars[n_vars + n_slack:].tolist() if n_artificial > 0 else []
        
        return {
            'status': 'optimal',
            'solution': self.solution.tolist(),
            'optimal_value': float(self.optimal_value),
            'slack_variables': slack_vars,
            'artificial_variables': artificial_vars,
            'iterations': self.iterations,
            'method': 'Big M Method'
        }
    
    def _build_initial_tableau(self, M: float) -> Tuple[np.ndarray, List[int]]:
        """Construye la tabla inicial para el Método de la Gran M"""
        n_vars = len(self.c_original)
        n_constraints = len(self.b_original)
        
        # Convertir a minimización si es necesario
        c = self.c_original.copy()
        if self.optimize == 'max':
            c = -c
        
        # Contar variables de holgura y artificiales necesarias
        n_slack = 0
        n_artificial = 0
        
        for constraint in self.constraints:
            if constraint == '<=':
                n_slack += 1
            elif constraint == '>=':
                n_slack += 1
                n_artificial += 1
            elif constraint == '=':
                n_artificial += 1
        
        total_vars = n_vars + n_slack + n_artificial
        
        # Crear tabla (n_constraints + 1 filas, total_vars + 1 columnas)
        tableau = np.zeros((n_constraints + 1, total_vars + 1))
        
        # Llenar coeficientes de restricciones
        tableau[:-1, :n_vars] = self.A_original
        
        # Añadir términos independientes
        tableau[:-1, -1] = self.b_original
        
        # Añadir variables de holgura y artificiales
        slack_idx = n_vars
        artificial_idx = n_vars + n_slack
        artificial_vars = []
        
        for i, constraint in enumerate(self.constraints):
            if constraint == '<=':
                tableau[i, slack_idx] = 1
                slack_idx += 1
            elif constraint == '>=':
                tableau[i, slack_idx] = -1
                tableau[i, artificial_idx] = 1
                artificial_vars.append(artificial_idx)
                slack_idx += 1
                artificial_idx += 1
            elif constraint == '=':
                tableau[i, artificial_idx] = 1
                artificial_vars.append(artificial_idx)
                artificial_idx += 1
        
        # Configurar función objetivo
        tableau[-1, :n_vars] = c
        
        # Añadir penalización M a variables artificiales
        for art_var in artificial_vars:
            tableau[-1, art_var] = M
        
        # Ajustar la fila objetivo para eliminar variables artificiales de la base
        for art_var in artificial_vars:
            for i in range(n_constraints):
                if abs(tableau[i, art_var] - 1) < 1e-10:
                    tableau[-1] = tableau[-1] - M * tableau[i]
        
        return tableau, artificial_vars


class TwoPhaseMethod(SimplexSolver):
    """Método de las Dos Fases (Two Phase Method)"""
    
    def solve(self) -> Dict:
        """Resuelve el problema usando el Método de las Dos Fases"""
        self.iterations = []
        n_vars = len(self.c_original)
        
        # FASE 1: Encontrar una solución básica factible
        phase1_result = self._phase1()
        
        if phase1_result['status'] == 'infeasible':
            return phase1_result
        
        # FASE 2: Optimizar la función objetivo original
        phase2_result = self._phase2(phase1_result)
        
        return phase2_result
    
    def _phase1(self) -> Dict:
        """Fase 1: Minimizar la suma de variables artificiales"""
        n_vars = len(self.c_original)
        n_constraints = len(self.b_original)
        
        # Construir tabla para fase 1
        tableau, artificial_vars, n_slack = self._build_phase1_tableau()
        
        self.iterations.append({
            'iteration': 0,
            'phase': 1,
            'tableau': tableau.copy(),
            'description': 'Fase 1: Tabla inicial para encontrar solución básica factible'
        })
        
        # Resolver fase 1 (minimizar suma de artificiales)
        iteration = 1
        max_iterations = 100
        
        while not self._is_optimal_phase1(tableau) and iteration < max_iterations:
            # Encontrar columna pivote
            pivot_col = self._find_pivot_column_phase1(tableau)
            if pivot_col is None:
                break
            
            # Encontrar fila pivote
            pivot_row = self._find_pivot_row(tableau, pivot_col)
            if pivot_row is None:
                return {
                    'status': 'unbounded',
                    'message': 'Problema no acotado en Fase 1',
                    'iterations': self.iterations
                }
            
            # Realizar pivote
            tableau = self._pivot(tableau, pivot_row, pivot_col)
            
            self.iterations.append({
                'iteration': iteration,
                'phase': 1,
                'tableau': tableau.copy(),
                'pivot_row': pivot_row,
                'pivot_col': pivot_col,
                'entering_row': pivot_row,
                'description': f'Fase 1 - Pivote en fila {pivot_row + 1}, columna {pivot_col + 1}'
            })
            
            iteration += 1
        
        # Verificar si se encontró una solución factible
        if abs(tableau[-1, -1]) > 1e-10:
            return {
                'status': 'infeasible',
                'message': 'El problema no tiene solución factible',
                'iterations': self.iterations
            }
        
        self.iterations.append({
            'iteration': iteration,
            'phase': 1,
            'tableau': tableau.copy(),
            'description': 'Fase 1 completada: Se encontró solución básica factible'
        })
        
        return {
            'status': 'feasible',
            'tableau': tableau,
            'artificial_vars': artificial_vars,
            'n_slack': n_slack
        }
    
    def _phase2(self, phase1_result: Dict) -> Dict:
        """Fase 2: Optimizar la función objetivo original"""
        n_vars = len(self.c_original)
        n_constraints = len(self.b_original)
        
        # Construir tabla para fase 2 a partir del resultado de fase 1
        # Eliminar columnas de variables artificiales
        tableau = self._build_phase2_tableau(phase1_result)
        
        self.iterations.append({
            'iteration': 0,
            'phase': 2,
            'tableau': tableau.copy(),
            'description': 'Fase 2: Tabla inicial para optimizar función objetivo'
        })
        
        # Resolver fase 2
        iteration = 1
        max_iterations = 100
        
        while not self._is_optimal(tableau) and iteration < max_iterations:
            # Encontrar columna pivote
            pivot_col = self._find_pivot_column(tableau)
            if pivot_col is None:
                break
            
            # Encontrar fila pivote
            pivot_row = self._find_pivot_row(tableau, pivot_col)
            if pivot_row is None:
                return {
                    'status': 'unbounded',
                    'message': 'El problema no está acotado',
                    'iterations': self.iterations
                }
            
            # Realizar pivote
            tableau = self._pivot(tableau, pivot_row, pivot_col)
            
            self.iterations.append({
                'iteration': iteration,
                'phase': 2,
                'tableau': tableau.copy(),
                'pivot_row': pivot_row,
                'pivot_col': pivot_col,
                'entering_row': pivot_row,
                'description': f'Fase 2 - Pivote en fila {pivot_row + 1}, columna {pivot_col + 1}'
            })
            
            iteration += 1
        
        if iteration >= max_iterations:
            return {
                'status': 'error',
                'message': 'Se alcanzó el número máximo de iteraciones',
                'iterations': self.iterations
            }
        
        # Extraer solución
        self.solution, self.optimal_value = self._extract_solution(tableau, n_vars)
        all_vars = self._extract_all_variables(tableau)
        self.status = 'optimal'
        
        # Separar variables por tipo (en fase 2 ya no hay artificiales)
        n_slack = sum(1 for c in self.constraints if c in ['<=', '>='])
        slack_vars = all_vars[n_vars:n_vars + n_slack].tolist() if n_slack > 0 else []
        
        return {
            'status': 'optimal',
            'solution': self.solution.tolist(),
            'optimal_value': float(self.optimal_value),
            'slack_variables': slack_vars,
            'artificial_variables': [],  # Ya fueron eliminadas en fase 2
            'iterations': self.iterations,
            'method': 'Two Phase Method'
        }
    
    def _build_phase1_tableau(self) -> Tuple[np.ndarray, List[int], int]:
        """Construye la tabla para la Fase 1"""
        n_vars = len(self.c_original)
        n_constraints = len(self.b_original)
        
        # Contar variables necesarias
        n_slack = 0
        n_artificial = 0
        
        for constraint in self.constraints:
            if constraint == '<=':
                n_slack += 1
            elif constraint == '>=':
                n_slack += 1
                n_artificial += 1
            elif constraint == '=':
                n_artificial += 1
        
        total_vars = n_vars + n_slack + n_artificial
        
        # Crear tabla
        tableau = np.zeros((n_constraints + 1, total_vars + 1))
        
        # Llenar coeficientes
        tableau[:-1, :n_vars] = self.A_original
        tableau[:-1, -1] = self.b_original
        
        # Añadir variables de holgura y artificiales
        slack_idx = n_vars
        artificial_idx = n_vars + n_slack
        artificial_vars = []
        
        for i, constraint in enumerate(self.constraints):
            if constraint == '<=':
                tableau[i, slack_idx] = 1
                slack_idx += 1
            elif constraint == '>=':
                tableau[i, slack_idx] = -1
                tableau[i, artificial_idx] = 1
                artificial_vars.append(artificial_idx)
                slack_idx += 1
                artificial_idx += 1
            elif constraint == '=':
                tableau[i, artificial_idx] = 1
                artificial_vars.append(artificial_idx)
                artificial_idx += 1
        
        # Función objetivo de Fase 1: minimizar suma de artificiales
        for art_var in artificial_vars:
            tableau[-1, art_var] = 1
        
        # Ajustar fila objetivo
        for art_var in artificial_vars:
            for i in range(n_constraints):
                if abs(tableau[i, art_var] - 1) < 1e-10:
                    tableau[-1] = tableau[-1] - tableau[i]
        
        return tableau, artificial_vars, n_slack
    
    def _build_phase2_tableau(self, phase1_result: Dict) -> np.ndarray:
        """Construye la tabla para la Fase 2 a partir del resultado de Fase 1"""
        n_vars = len(self.c_original)
        n_constraints = len(self.b_original)
        phase1_tableau = phase1_result['tableau']
        artificial_vars = phase1_result['artificial_vars']
        n_slack = phase1_result['n_slack']
        
        # Eliminar columnas de variables artificiales
        # Mantener solo: variables originales + variables de holgura + RHS
        cols_to_keep = list(range(n_vars + n_slack)) + [phase1_tableau.shape[1] - 1]
        
        # Crear nueva tabla sin columnas artificiales
        tableau = np.zeros((n_constraints + 1, len(cols_to_keep)))
        for new_idx, old_idx in enumerate(cols_to_keep):
            tableau[:, new_idx] = phase1_tableau[:, old_idx]
        
        # Reiniciar fila objetivo con la función objetivo original
        c = self.c_original.copy()
        if self.optimize == 'max':
            c = -c
        
        tableau[-1, :n_vars] = c
        tableau[-1, n_vars:-1] = 0  # Coeficientes de variables de holgura
        tableau[-1, -1] = 0  # RHS
        
        # Ajustar fila objetivo para todas las variables básicas
        for j in range(tableau.shape[1] - 1):
            col = tableau[:-1, j]
            # Si esta columna es básica (tiene un 1 y resto 0s)
            if np.count_nonzero(np.abs(col - 1) < 1e-10) == 1 and np.count_nonzero(np.abs(col) < 1e-10) == len(col) - 1:
                idx = np.argmax(np.abs(col - 1) < 1e-10)
                # Si el coeficiente en la fila objetivo no es cero, ajustar
                if abs(tableau[-1, j]) > 1e-10:
                    factor = tableau[-1, j]
                    tableau[-1] = tableau[-1] - factor * tableau[idx]
        
        return tableau
    
    def _is_optimal_phase1(self, tableau: np.ndarray) -> bool:
        """Verifica si la tabla de Fase 1 es óptima"""
        return np.all(tableau[-1, :-1] >= -1e-10)
    
    def _find_pivot_column_phase1(self, tableau: np.ndarray) -> Optional[int]:
        """Encuentra la columna pivote en Fase 1"""
        obj_row = tableau[-1, :-1]
        if np.all(obj_row >= -1e-10):
            return None
        return np.argmin(obj_row)
