"""
Aplicación web Flask para resolver problemas de optimización lineal
"""

from flask import Flask, render_template, request, jsonify
from simplex import BigMMethod, TwoPhaseMethod
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@app.route('/solve', methods=['POST'])
def solve():
    """Endpoint para resolver problemas de optimización"""
    try:
        data = request.json
        
        # Extraer datos del problema
        c = data['objective']
        A = data['constraints_matrix']
        b = data['rhs']
        constraints = data['constraint_types']
        optimize = data['optimize']
        method = data['method']
        
        # Validar datos
        if not c or not A or not b or not constraints:
            return jsonify({
                'status': 'error',
                'message': 'Datos incompletos'
            }), 400
        
        # Resolver según el método seleccionado
        if method == 'bigm':
            solver = BigMMethod(c, A, b, constraints, optimize)
            M = data.get('M', 1000.0)
            result = solver.solve(M=M)
        elif method == 'twophase':
            solver = TwoPhaseMethod(c, A, b, constraints, optimize)
            result = solver.solve()
        else:
            return jsonify({
                'status': 'error',
                'message': 'Método no válido'
            }), 400
        
        # Formatear iteraciones para la respuesta
        formatted_iterations = []
        for iter_data in result.get('iterations', []):
            formatted_iter = {
                'iteration': int(iter_data.get('iteration', 0)),
                'phase': int(iter_data.get('phase')) if iter_data.get('phase') is not None else None,
                'description': str(iter_data.get('description', '')),
                'tableau': iter_data['tableau'].tolist() if 'tableau' in iter_data else None,
                'pivot_row': int(iter_data.get('pivot_row')) if iter_data.get('pivot_row') is not None else None,
                'pivot_col': int(iter_data.get('pivot_col')) if iter_data.get('pivot_col') is not None else None,
                'entering_row': int(iter_data.get('entering_row')) if iter_data.get('entering_row') is not None else None
            }
            # Añadir versión formateada de la tabla en HTML
            if 'tableau' in iter_data:
                try:
                    tbl = iter_data['tableau']
                    if not hasattr(tbl, 'shape'):
                        tbl = np.array(tbl, dtype=float)
                    # Pasar información del pivote
                    pivot_row = iter_data.get('pivot_row')
                    pivot_col = iter_data.get('pivot_col')
                    entering_row = iter_data.get('entering_row')
                    formatted_iter['tableau_html'] = solver.format_tableau_html(
                        tbl, 
                        cj=solver.c_original,
                        pivot_row=pivot_row,
                        pivot_col=pivot_col,
                        entering_row=entering_row
                    )
                except Exception:
                    formatted_iter['tableau_html'] = None

            formatted_iterations.append(formatted_iter)
        
        result['iterations'] = formatted_iterations
        
        # Ensure all numeric values are Python types, not numpy types
        if 'optimal_value' in result:
            result['optimal_value'] = float(result['optimal_value'])
        if 'solution' in result and result['solution']:
            result['solution'] = [float(x) for x in result['solution']]
        
        return jsonify(result)
    
    except Exception as e:
        # Log the error for debugging but don't expose stack traces
        import logging
        logging.error(f'Error solving optimization problem: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': 'Error al resolver el problema. Por favor, verifica los datos ingresados.'
        }), 500


@app.route('/examples')
def examples():
    """Devuelve ejemplos de problemas"""
    examples_list = [
        {
            'name': 'Problema de Maximización Simple',
            'description': 'Maximizar z = 3x₁ + 2x₂',
            'objective': [3, 2],
            'constraints_matrix': [
                [2, 1],
                [1, 2],
                [1, 0]
            ],
            'rhs': [18, 12, 5],
            'constraint_types': ['<=', '<=', '<='],
            'optimize': 'max'
        },
        {
            'name': 'Problema con Restricciones Mixtas',
            'description': 'Maximizar z = 5x₁ + 4x₂',
            'objective': [5, 4],
            'constraints_matrix': [
                [1, 1],
                [2, 1],
                [1, 2]
            ],
            'rhs': [5, 8, 7],
            'constraint_types': ['>=', '<=', '<='],
            'optimize': 'max'
        },
        {
            'name': 'Problema de Minimización',
            'description': 'Minimizar z = 2x₁ + 3x₂',
            'objective': [2, 3],
            'constraints_matrix': [
                [1, 1],
                [2, 1],
                [1, 3]
            ],
            'rhs': [4, 6, 6],
            'constraint_types': ['>=', '>=', '>='],
            'optimize': 'min'
        },
        {
            'name': 'Problema de Producción',
            'description': 'Una fábrica produce dos productos. Maximizar ganancia.',
            'objective': [40, 30],
            'constraints_matrix': [
                [1, 1],
                [2, 1],
                [1, 2]
            ],
            'rhs': [12, 16, 15],
            'constraint_types': ['<=', '<=', '<='],
            'optimize': 'max'
        }
    ]
    
    return jsonify(examples_list)


if __name__ == '__main__':
    import os
    # Only enable debug mode in development environment
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
