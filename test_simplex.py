"""
Script de prueba para verificar el funcionamiento de los algoritmos
"""

from simplex import BigMMethod, TwoPhaseMethod


def test_example_1():
    """Problema de maximización simple"""
    print("\n" + "="*60)
    print("Prueba 1: Problema de Maximización Simple")
    print("="*60)
    print("Maximizar: z = 3x₁ + 2x₂")
    print("Sujeto a:")
    print("  2x₁ + x₂ ≤ 18")
    print("  x₁ + 2x₂ ≤ 12")
    print("  x₁ ≤ 5")
    print("  x₁, x₂ ≥ 0")
    
    c = [3, 2]
    A = [[2, 1], [1, 2], [1, 0]]
    b = [18, 12, 5]
    constraints = ['<=', '<=', '<=']
    
    # Método Gran M
    print("\n--- Método de la Gran M ---")
    solver_bigm = BigMMethod(c, A, b, constraints, 'max')
    result_bigm = solver_bigm.solve(M=1000)
    print(f"Estado: {result_bigm['status']}")
    if result_bigm['status'] == 'optimal':
        print(f"Valor óptimo: {result_bigm['optimal_value']:.4f}")
        print(f"Solución: x₁ = {result_bigm['solution'][0]:.4f}, x₂ = {result_bigm['solution'][1]:.4f}")
        print(f"Iteraciones: {len(result_bigm['iterations'])}")
        # Imprimir tablas formateadas de cada iteración
        for it in solver_bigm.iterations:
            print(f"\nIteración {it.get('iteration', '?')}: {it.get('description', '')}")
            try:
                print(solver_bigm.format_tableau(it['tableau'], cj=solver_bigm.c_original))
            except Exception as e:
                print(f"Error al formatear la tabla: {e}")
    
    # Método Dos Fases
    print("\n--- Método de las Dos Fases ---")
    solver_twophase = TwoPhaseMethod(c, A, b, constraints, 'max')
    result_twophase = solver_twophase.solve()
    print(f"Estado: {result_twophase['status']}")
    if result_twophase['status'] == 'optimal':
        print(f"Valor óptimo: {result_twophase['optimal_value']:.4f}")
        print(f"Solución: x₁ = {result_twophase['solution'][0]:.4f}, x₂ = {result_twophase['solution'][1]:.4f}")
        print(f"Iteraciones: {len(result_twophase['iterations'])}")
        for it in solver_twophase.iterations:
            print(f"\nIteración {it.get('iteration', '?')} (fase {it.get('phase','?')}): {it.get('description', '')}")
            try:
                # para TwoPhase, pasar c_original para mostrar CJ
                print(solver_twophase.format_tableau(it['tableau'], cj=solver_twophase.c_original))
            except Exception as e:
                print(f"Error al formatear la tabla: {e}")
    
    return result_bigm['status'] == 'optimal' and result_twophase['status'] == 'optimal'


def test_example_2():
    """Problema con restricciones mixtas"""
    print("\n" + "="*60)
    print("Prueba 2: Problema con Restricciones Mixtas")
    print("="*60)
    print("Maximizar: z = 5x₁ + 4x₂")
    print("Sujeto a:")
    print("  x₁ + x₂ ≥ 5")
    print("  2x₁ + x₂ ≤ 8")
    print("  x₁ + 2x₂ ≤ 7")
    print("  x₁, x₂ ≥ 0")
    
    c = [5, 4]
    A = [[1, 1], [2, 1], [1, 2]]
    b = [5, 8, 7]
    constraints = ['>=', '<=', '<=']
    
    # Método Gran M
    print("\n--- Método de la Gran M ---")
    solver_bigm = BigMMethod(c, A, b, constraints, 'max')
    result_bigm = solver_bigm.solve(M=1000)
    print(f"Estado: {result_bigm['status']}")
    if result_bigm['status'] == 'optimal':
        print(f"Valor óptimo: {result_bigm['optimal_value']:.4f}")
        print(f"Solución: x₁ = {result_bigm['solution'][0]:.4f}, x₂ = {result_bigm['solution'][1]:.4f}")
        print(f"Iteraciones: {len(result_bigm['iterations'])}")
    
    # Método Dos Fases
    print("\n--- Método de las Dos Fases ---")
    solver_twophase = TwoPhaseMethod(c, A, b, constraints, 'max')
    result_twophase = solver_twophase.solve()
    print(f"Estado: {result_twophase['status']}")
    if result_twophase['status'] == 'optimal':
        print(f"Valor óptimo: {result_twophase['optimal_value']:.4f}")
        print(f"Solución: x₁ = {result_twophase['solution'][0]:.4f}, x₂ = {result_twophase['solution'][1]:.4f}")
        print(f"Iteraciones: {len(result_twophase['iterations'])}")
    
    return result_bigm['status'] == 'optimal' and result_twophase['status'] == 'optimal'


def test_example_3():
    """Problema de minimización"""
    print("\n" + "="*60)
    print("Prueba 3: Problema de Minimización")
    print("="*60)
    print("Minimizar: z = 2x₁ + 3x₂")
    print("Sujeto a:")
    print("  x₁ + x₂ ≥ 4")
    print("  2x₁ + x₂ ≥ 6")
    print("  x₁ + 3x₂ ≥ 6")
    print("  x₁, x₂ ≥ 0")
    
    c = [2, 3]
    A = [[1, 1], [2, 1], [1, 3]]
    b = [4, 6, 6]
    constraints = ['>=', '>=', '>=']
    
    # Método Gran M
    print("\n--- Método de la Gran M ---")
    solver_bigm = BigMMethod(c, A, b, constraints, 'min')
    result_bigm = solver_bigm.solve(M=1000)
    print(f"Estado: {result_bigm['status']}")
    if result_bigm['status'] == 'optimal':
        print(f"Valor óptimo: {result_bigm['optimal_value']:.4f}")
        print(f"Solución: x₁ = {result_bigm['solution'][0]:.4f}, x₂ = {result_bigm['solution'][1]:.4f}")
        print(f"Iteraciones: {len(result_bigm['iterations'])}")
    
    # Método Dos Fases
    print("\n--- Método de las Dos Fases ---")
    solver_twophase = TwoPhaseMethod(c, A, b, constraints, 'min')
    result_twophase = solver_twophase.solve()
    print(f"Estado: {result_twophase['status']}")
    if result_twophase['status'] == 'optimal':
        print(f"Valor óptimo: {result_twophase['optimal_value']:.4f}")
        print(f"Solución: x₁ = {result_twophase['solution'][0]:.4f}, x₂ = {result_twophase['solution'][1]:.4f}")
        print(f"Iteraciones: {len(result_twophase['iterations'])}")
    
    return result_bigm['status'] == 'optimal' and result_twophase['status'] == 'optimal'


def main():
    """Ejecutar todas las pruebas"""
    print("\n" + "#"*60)
    print("# PRUEBAS DEL SISTEMA DE OPTIMIZACIÓN LINEAL")
    print("#"*60)
    
    results = []
    
    try:
        results.append(("Prueba 1", test_example_1()))
    except Exception as e:
        print(f"\nError en Prueba 1: {e}")
        results.append(("Prueba 1", False))
    
    try:
        results.append(("Prueba 2", test_example_2()))
    except Exception as e:
        print(f"\nError en Prueba 2: {e}")
        results.append(("Prueba 2", False))
    
    try:
        results.append(("Prueba 3", test_example_3()))
    except Exception as e:
        print(f"\nError en Prueba 3: {e}")
        results.append(("Prueba 3", False))
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE PRUEBAS")
    print("="*60)
    for name, result in results:
        status = "✓ PASÓ" if result else "✗ FALLÓ"
        print(f"{name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    print(f"\nTotal: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
        return True
    else:
        print(f"\n⚠️ {total - passed} prueba(s) fallaron")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
