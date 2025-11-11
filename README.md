# Sistema Interactivo de Optimización Lineal

Proyecto de Investigación de Operaciones I - Software capaz de resolver problemas de optimización lineal utilizando los métodos de **Gran M (Big M)** y **Dos Fases (Two Phase)**.

## 🎯 Características

- **Interfaz Web Interactiva**: Aplicación web moderna y fácil de usar
- **Dos Métodos de Solución**:
  - Método de la Gran M (Big M Method)
  - Método de las Dos Fases (Two Phase Method)
- **Visualización Completa**: Muestra todas las iteraciones del algoritmo Simplex
- **Ejemplos Predefinidos**: Incluye problemas de ejemplo para aprender
- **Soporte para Múltiples Tipos de Restricciones**: ≤, ≥, =
- **Maximización y Minimización**: Resuelve ambos tipos de problemas

## 📋 Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## 🚀 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/Elgona9/proyectoIO1.git
cd proyectoIO1
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 💻 Uso

1. Iniciar la aplicación:
```bash
python app.py
```

2. Abrir el navegador web en:
```
http://localhost:5000
```

3. Usar la interfaz para:
   - Seleccionar un ejemplo predefinido
   - Configurar un problema personalizado
   - Elegir el método de solución (Gran M o Dos Fases)
   - Ver los resultados y las iteraciones paso a paso

## 📊 Ejemplos de Problemas

### Ejemplo 1: Problema de Maximización Simple
```
Maximizar: z = 3x₁ + 2x₂
Sujeto a:
  2x₁ + x₂ ≤ 18
  x₁ + 2x₂ ≤ 12
  x₁ ≤ 5
  x₁, x₂ ≥ 0
```

### Ejemplo 2: Problema con Restricciones Mixtas
```
Maximizar: z = 5x₁ + 4x₂
Sujeto a:
  x₁ + x₂ ≥ 5
  2x₁ + x₂ ≤ 8
  x₁ + 2x₂ ≤ 7
  x₁, x₂ ≥ 0
```

## 🔬 Métodos Implementados

### Método de la Gran M (Big M)
El método de la Gran M es una técnica para resolver problemas de programación lineal que contienen restricciones de igualdad o de mayor-igual. Utiliza una constante M muy grande para penalizar variables artificiales en la función objetivo.

**Ventajas:**
- Método directo de una sola fase
- Fácil de implementar
- Conceptualmente simple

**Desventajas:**
- Puede tener problemas numéricos si M es muy grande
- Menos estable computacionalmente que el método de dos fases

### Método de las Dos Fases (Two Phase)
El método de las dos fases resuelve el problema en dos etapas:

**Fase 1:** Encuentra una solución básica factible inicial minimizando la suma de variables artificiales.

**Fase 2:** Una vez encontrada una solución factible, optimiza la función objetivo original.

**Ventajas:**
- Más estable numéricamente
- No requiere elegir un valor de M
- Claramente indica si el problema es infactible

**Desventajas:**
- Requiere dos fases de optimización
- Puede ser más lento en algunos casos

## 🏗️ Estructura del Proyecto

```
proyectoIO1/
├── app.py              # Aplicación Flask (servidor web)
├── simplex.py          # Implementación de algoritmos
├── templates/
│   └── index.html      # Interfaz web
├── requirements.txt    # Dependencias
└── README.md          # Documentación
```

## 🧮 Formulación Matemática

### Forma Estándar de un Problema de Programación Lineal

**Maximizar/Minimizar:** z = c₁x₁ + c₂x₂ + ... + cₙxₙ

**Sujeto a:**
- a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ {≤,≥,=} b₁
- a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ {≤,≥,=} b₂
- ...
- aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ {≤,≥,=} bₘ
- x₁, x₂, ..., xₙ ≥ 0

## 🎓 Conceptos Teóricos

### Variables de Holgura (Slack Variables)
Se añaden a restricciones del tipo ≤ para convertirlas en igualdades.

### Variables de Exceso (Surplus Variables)
Se restan de restricciones del tipo ≥ para convertirlas en igualdades.

### Variables Artificiales
Se añaden temporalmente para obtener una solución básica factible inicial.

### Tabla Simplex
Matriz que contiene los coeficientes del sistema de ecuaciones en cada iteración.

### Criterio de Optimalidad
- **Maximización**: Todos los coeficientes en la fila objetivo deben ser ≤ 0
- **Minimización**: Todos los coeficientes en la fila objetivo deben ser ≥ 0

## 📚 Referencias

- Taha, H. A. (2017). *Investigación de Operaciones* (10ª ed.). Pearson.
- Hillier, F. S., & Lieberman, G. J. (2015). *Introducción a la Investigación de Operaciones* (10ª ed.). McGraw-Hill.
- Winston, W. L. (2004). *Investigación de Operaciones: Aplicaciones y Algoritmos* (4ª ed.). Thomson.

## 👥 Autor

Proyecto desarrollado para el curso de Investigación de Operaciones I.

## 📄 Licencia

Este proyecto es de código abierto y está disponible para fines educativos.
