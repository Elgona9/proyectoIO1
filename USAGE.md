# Guía de Uso - Sistema de Optimización Lineal

## Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/Elgona9/proyectoIO1.git
cd proyectoIO1
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Ejecución

### Modo Desarrollo
```bash
FLASK_ENV=development python app.py
```

### Modo Producción
```bash
python app.py
```

La aplicación estará disponible en: http://localhost:5000

## Uso de la Interfaz Web

### 1. Seleccionar un Ejemplo Predefinido
- Haga clic en cualquiera de los ejemplos en la sección "Ejemplos Predefinidos"
- El problema se cargará automáticamente en los formularios

### 2. Configurar un Problema Personalizado

#### Configuración del Problema
- **Tipo de Optimización**: Seleccione "Maximizar" o "Minimizar"
- **Método de Solución**: 
  - "Método de la Gran M" - Usa penalización M para variables artificiales
  - "Método de las Dos Fases" - Resuelve en dos fases separadas
- **Valor de M**: Solo para método Gran M (valor por defecto: 1000)

#### Función Objetivo
1. Especifique el número de variables
2. Haga clic en "Actualizar"
3. Ingrese los coeficientes de cada variable

#### Restricciones
1. Especifique el número de restricciones
2. Haga clic en "Actualizar"
3. Para cada restricción:
   - Ingrese los coeficientes de las variables
   - Seleccione el tipo de restricción (≤, ≥, =)
   - Ingrese el valor del lado derecho

### 3. Resolver el Problema
- Haga clic en "🚀 Resolver Problema"
- Los resultados se mostrarán debajo del formulario

### 4. Interpretar los Resultados

#### Solución Óptima
- **Valor óptimo**: El valor de la función objetivo en el punto óptimo
- **Variables de decisión**: Los valores de x₁, x₂, etc. en la solución óptima

#### Iteraciones del Algoritmo
- Cada iteración muestra:
  - La tabla Simplex en ese momento
  - Descripción de la operación realizada
  - Elemento pivote resaltado en amarillo

## Uso desde Línea de Comandos

### Ejecutar Tests
```bash
python test_simplex.py
```

### Ejemplo de Uso Programático
```python
from simplex import BigMMethod, TwoPhaseMethod

# Definir el problema
c = [3, 2]  # Coeficientes función objetivo
A = [[2, 1], [1, 2], [1, 0]]  # Matriz de restricciones
b = [18, 12, 5]  # Lado derecho
constraints = ['<=', '<=', '<=']  # Tipos de restricciones

# Método Gran M
solver = BigMMethod(c, A, b, constraints, 'max')
result = solver.solve(M=1000)
print(f"Solución: {result['solution']}")
print(f"Valor óptimo: {result['optimal_value']}")

# Método Dos Fases
solver = TwoPhaseMethod(c, A, b, constraints, 'max')
result = solver.solve()
print(f"Solución: {result['solution']}")
print(f"Valor óptimo: {result['optimal_value']}")
```

## Ejemplos de Problemas

### Ejemplo 1: Maximización Simple
```
Maximizar: z = 3x₁ + 2x₂
Sujeto a:
  2x₁ + x₂ ≤ 18
  x₁ + 2x₂ ≤ 12
  x₁ ≤ 5
  x₁, x₂ ≥ 0

Solución: x₁ = 5, x₂ = 3.5, z = 22
```

### Ejemplo 2: Problema con Restricciones Mixtas
```
Maximizar: z = 5x₁ + 4x₂
Sujeto a:
  x₁ + x₂ ≥ 5
  2x₁ + x₂ ≤ 8
  x₁ + 2x₂ ≤ 7
  x₁, x₂ ≥ 0

Solución: x₁ = 3, x₂ = 2, z = 23
```

### Ejemplo 3: Minimización
```
Minimizar: z = 2x₁ + 3x₂
Sujeto a:
  x₁ + x₂ ≥ 4
  2x₁ + x₂ ≥ 6
  x₁ + 3x₂ ≥ 6
  x₁, x₂ ≥ 0

Solución: x₁ = 3, x₂ = 1, z = 9
```

## Solución de Problemas

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Error: "Address already in use"
```bash
# Encontrar y detener el proceso en el puerto 5000
lsof -i :5000
kill -9 <PID>
```

### La página no carga
- Verifique que el servidor esté ejecutándose
- Asegúrese de estar accediendo a http://localhost:5000
- Revise los logs en la consola donde ejecutó `python app.py`

## Recursos Adicionales

- [README.md](README.md) - Documentación completa
- [test_simplex.py](test_simplex.py) - Ejemplos de uso de las clases
- [simplex.py](simplex.py) - Código fuente de los algoritmos

## Soporte

Para reportar problemas o sugerir mejoras, por favor abra un issue en el repositorio de GitHub.
