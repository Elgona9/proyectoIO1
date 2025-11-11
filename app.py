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
                'iteration': iter_data.get('iteration', 0),
                'phase': iter_data.get('phase', None),
                'description': iter_data.get('description', ''),
                'tableau': iter_data['tableau'].tolist() if 'tableau' in iter_data else None,
                'pivot_row': iter_data.get('pivot_row', None),
                'pivot_col': iter_data.get('pivot_col', None)
            }
            formatted_iterations.append(formatted_iter)
        
        result['iterations'] = formatted_iterations
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error al resolver el problema: {str(e)}'
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
    app.run(debug=True, host='0.0.0.0', port=5000)
