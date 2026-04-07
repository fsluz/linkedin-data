// ============================================
// ML Dashboard - FastAPI Integration
// ============================================

class MLDashboard {
  constructor() {
    this.apiUrl = 'http://localhost:8000';
    this.charts = {};
    this.data = {};
    this.mockMode = true;

    this.theme = this.getTheme();
    this.applyChartDefaults();
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.updateContextUrl();
    this.checkAPIConnection();
    this.loadMockData();
    this.setupTabs();
  }

  getTheme() {
    const styles = getComputedStyle(document.documentElement);
    const read = (name, fallback) => styles.getPropertyValue(name).trim() || fallback;
    return {
      text: read('--chart-text', '#e7edf7'),
      muted: read('--chart-muted', '#a4b2c8'),
      grid: read('--chart-grid', '#22324c'),
      border: read('--chart-border', '#0c1322'),
      c1: read('--chart-1', '#4ea3ff'),
      c2: read('--chart-2', '#f2c811'),
      c3: read('--chart-3', '#2dd4bf'),
      c4: read('--chart-4', '#f97316'),
      c5: read('--chart-5', '#a78bfa'),
      success: read('--success', '#2dd4bf'),
      danger: read('--danger', '#f87171')
    };
  }

  applyChartDefaults() {
    if (!window.Chart) return;
    Chart.defaults.color = this.theme.text;
    Chart.defaults.borderColor = this.theme.grid;
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  }

  withAlpha(hex, alpha) {
    const normalized = hex.replace('#', '');
    const value = normalized.length === 3
      ? normalized.split('').map((c) => c + c).join('')
      : normalized;
    const intVal = parseInt(value, 16);
    const r = (intVal >> 16) & 255;
    const g = (intVal >> 8) & 255;
    const b = intVal & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  setupEventListeners() {
    // Filters
    document.getElementById('btn-refresh').addEventListener('click', () => this.refreshDashboard());
    document.getElementById('btn-export').addEventListener('click', () => this.exportData());
    document.getElementById('api-url').addEventListener('change', (e) => {
      this.apiUrl = e.target.value;
      this.updateContextUrl();
      this.checkAPIConnection();
    });

    // Classification
    document.getElementById('btn-clf-predict').addEventListener('click', () => this.predictClassification());

    // Regression
    document.getElementById('btn-reg-predict').addEventListener('click', () => this.predictRegression());

    // Recommendation
    document.getElementById('btn-rec-predict').addEventListener('click', () => this.predictRecommendation());
  }

  setupTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', (e) => {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
          section.style.display = 'none';
        });

        // Remove active from all tabs
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));

        // Show selected section
        const tabName = e.target.getAttribute('data-tab');
        const section = document.getElementById(tabName);
        if (section) section.style.display = 'block';

        // Add active to clicked tab
        e.target.classList.add('active');

        // Initialize charts if needed
        setTimeout(() => {
          if (tabName === 'classification' && !this.charts.confusionMatrix) {
            this.initClassificationCharts();
          } else if (tabName === 'regression' && !this.charts.regressionScatter) {
            this.initRegressionCharts();
          } else if (tabName === 'clustering' && !this.charts.clusterDistribution) {
            this.initClusteringCharts();
          } else if (tabName === 'recommendation' && !this.charts.relevanceTrend) {
            this.initRecommendationCharts();
          }
        }, 100);
      });
    });
  }

  async checkAPIConnection() {
    try {
      const response = await fetch(`${this.apiUrl}/health`, {
        method: 'GET',
        mode: 'cors'
      });

      const statusDot = document.getElementById('api-status');
      const statusText = document.getElementById('api-status-text');

      if (response.ok) {
        statusDot.classList.remove('offline');
        statusText.textContent = 'API Conectada';
        this.mockMode = false;
        this.loadRealData();
      } else {
        throw new Error('API retornou erro');
      }
    } catch (error) {
      const statusDot = document.getElementById('api-status');
      const statusText = document.getElementById('api-status-text');
      statusDot.classList.add('offline');
      statusText.textContent = 'Modo Demo (desconectada)';
      this.mockMode = true;
      console.warn('API indisponível, usando dados de teste:', error);
    }
  }

  loadMockData() {
    this.data = {
      overview: {
        modelsCount: 4,
        accuracy: 0.87,
        processed: 5234,
        latency: 145
      },
      classification: {
        accuracy: 0.89,
        precision: 0.87,
        f1: 0.88,
        confusionMatrix: [[450, 50], [40, 460]]
      },
      regression: {
        rmse: 2.34,
        mae: 1.67,
        r2: 0.92
      },
      clustering: {
        k: 5,
        silhouette: 0.76,
        samples: 5234
      },
      recommendation: {
        hitRate: 0.84,
        latency: 234,
        coverage: 0.98
      }
    };

    this.renderOverview();
  }

  async loadRealData() {
    try {
      // Endpoints de exemplo - ajustar conforme sua API
      const classData = await this.apiCall('/ml/classification/metrics');
      const regData = await this.apiCall('/ml/regression/metrics');
      const clustData = await this.apiCall('/ml/clustering/metrics');
      const recData = await this.apiCall('/ml/recommendation/metrics');

      this.data = {
        classification: classData,
        regression: regData,
        clustering: clustData,
        recommendation: recData
      };

      this.renderOverview();
    } catch (error) {
      console.error('Erro ao carregar dados reais:', error);
      this.mockMode = true;
    }
  }

  async apiCall(endpoint, method = 'GET', body = null) {
    try {
      const options = {
        method,
        headers: {
          'Content-Type': 'application/json'
        },
        mode: 'cors'
      };

      if (body) options.body = JSON.stringify(body);

      const response = await fetch(`${this.apiUrl}${endpoint}`, options);

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      return await response.json();
    } catch (error) {
      console.error(`Erro na chamada ${endpoint}:`, error);
      throw error;
    }
  }

  renderOverview() {
    const data = this.data.overview || this.data.classification;

    document.getElementById('kpi-models-count').textContent = '4';
    document.getElementById('kpi-accuracy').textContent = (data.accuracy || 0.87).toLocaleString('pt-BR', {
      style: 'percent',
      minimumFractionDigits: 2
    });
    document.getElementById('kpi-processed').textContent = (data.processed || 5234).toLocaleString('pt-BR');
    document.getElementById('kpi-latency').textContent = (data.latency || 145).toFixed(0);

    // Initialize overview charts
    setTimeout(() => {
      this.initOverviewCharts();
      this.renderEndpointsTable();
    }, 100);
  }

  initOverviewCharts() {
    // Chart: Models Performance
    if (!this.charts.modelsPerformance) {
      const ctx = document.getElementById('chart-models-performance');
      if (ctx) {
        this.charts.modelsPerformance = new Chart(ctx, {
          type: 'radar',
          data: {
            labels: ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC'],
            datasets: [
              {
                label: 'Classificação',
                data: [89, 87, 88, 88, 91],
                borderColor: this.theme.c1,
                backgroundColor: this.withAlpha(this.theme.c1, 0.16),
                fill: true,
                tension: 0.4
              },
              {
                label: 'Regressão',
                data: [92, 90, 91, 91, 93],
                borderColor: this.theme.c2,
                backgroundColor: this.withAlpha(this.theme.c2, 0.16),
                fill: true,
                tension: 0.4
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: {
                labels: { color: this.theme.text }
              }
            },
            scales: {
              r: {
                beginAtZero: true,
                max: 100,
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              }
            }
          }
        });
      }
    }

    // Chart: Predictions Distribution
    if (!this.charts.predictionsDistribution) {
      const ctx = document.getElementById('chart-predictions-distribution');
      if (ctx) {
        this.charts.predictionsDistribution = new Chart(ctx, {
          type: 'doughnut',
          data: {
            labels: ['Classificação', 'Regressão', 'Clusterização', 'Recomendação'],
            datasets: [{
              data: [1250, 980, 1500, 1504],
              backgroundColor: [this.theme.c1, this.theme.c2, this.theme.success, this.theme.c4],
              borderColor: this.theme.border,
              borderWidth: 2
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: {
                position: 'bottom',
                labels: { color: this.theme.text, padding: 20 }
              }
            }
          }
        });
      }
    }
  }

  initClassificationCharts() {
    // Chart: Confusion Matrix
    if (!this.charts.confusionMatrix) {
      const ctx = document.getElementById('chart-confusion-matrix');
      if (ctx) {
        const data = this.data.classification || { confusionMatrix: [[450, 50], [40, 460]] };
        const cm = data.confusionMatrix;

        this.charts.confusionMatrix = new Chart(ctx, {
          type: 'bubble',
          data: {
            datasets: [
              {
                label: 'TN',
                data: [{ x: 0, y: 0, r: Math.sqrt(cm[0][0]) / 2 }],
                backgroundColor: this.withAlpha(this.theme.success, 0.6)
              },
              {
                label: 'FP',
                data: [{ x: 1, y: 0, r: Math.sqrt(cm[0][1]) / 2 }],
                backgroundColor: this.withAlpha(this.theme.danger, 0.6)
              },
              {
                label: 'FN',
                data: [{ x: 0, y: 1, r: Math.sqrt(cm[1][0]) / 2 }],
                backgroundColor: this.withAlpha(this.theme.danger, 0.6)
              },
              {
                label: 'TP',
                data: [{ x: 1, y: 1, r: Math.sqrt(cm[1][1]) / 2 }],
                backgroundColor: this.withAlpha(this.theme.success, 0.6)
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: { labels: { color: this.theme.text } }
            },
            scales: {
              x: {
                min: -0.5,
                max: 1.5,
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              },
              y: {
                min: -0.5,
                max: 1.5,
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              }
            }
          }
        });
      }
    }

    // Chart: Class Distribution
    if (!this.charts.classDistribution) {
      const ctx = document.getElementById('chart-class-distribution');
      if (ctx) {
        this.charts.classDistribution = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: ['Classe 0 (Sem Skill)', 'Classe 1 (Com Skill)'],
            datasets: [{
              label: 'Quantidade de Amostras',
              data: [2500, 2734],
              backgroundColor: [this.theme.c1, this.theme.c2],
              borderRadius: 6
            }]
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: { labels: { color: this.theme.text } }
            },
            scales: {
              x: {
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              },
              y: {
                ticks: { color: this.theme.muted }
              }
            }
          }
        });
      }
    }

    // Chart: Feature Importance
    if (!this.charts.featureImportance) {
      const ctx = document.getElementById('chart-feature-importance');
      if (ctx) {
        this.charts.featureImportance = new Chart(ctx, {
          type: 'barh',
          data: {
            labels: ['Python', 'JavaScript', 'SQL', 'React', 'Node.js', 'Docker', 'Kubernetes', 'AWS', 'Git', 'REST API'],
            datasets: [{
              label: 'Importância',
              data: [0.18, 0.16, 0.14, 0.12, 0.11, 0.09, 0.07, 0.06, 0.05, 0.02],
              backgroundColor: this.theme.c1,
              borderRadius: 4
            }]
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: { display: false }
            },
            scales: {
              x: {
                max: 0.2,
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              },
              y: {
                ticks: { color: this.theme.muted }
              }
            }
          }
        });
      }
    }

    // Populate metrics
    const clf = this.data.classification;
    document.getElementById('clf-accuracy').textContent = (clf.accuracy * 100).toFixed(1) + '%';
    document.getElementById('clf-precision').textContent = (clf.precision * 100).toFixed(1) + '%';
    document.getElementById('clf-f1').textContent = (clf.f1 * 100).toFixed(1) + '%';
  }

  initRegressionCharts() {
    // Chart: Regression Scatter
    if (!this.charts.regressionScatter) {
      const ctx = document.getElementById('chart-regression-scatter');
      if (ctx) {
        // Generate sample data
        const points = [];
        for (let i = 0; i < 100; i++) {
          points.push({
            x: Math.random() * 20,
            y: Math.random() * 20 + (Math.random() * 2 - 1)
          });
        }

        this.charts.regressionScatter = new Chart(ctx, {
          type: 'scatter',
          data: {
            datasets: [
              {
                label: 'Real vs Previsto',
                data: points,
                backgroundColor: this.withAlpha(this.theme.c1, 0.6),
                borderColor: this.theme.c1
              },
              {
                label: 'Linha Perfeita',
                type: 'line',
                data: [
                  { x: 0, y: 0 },
                  { x: 20, y: 20 }
                ],
                borderColor: this.theme.success,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: { labels: { color: this.theme.text } }
            },
            scales: {
              x: {
                title: { display: true, text: 'Valores Reais', color: this.theme.text },
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              },
              y: {
                title: { display: true, text: 'Valores Previstos', color: this.theme.text },
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              }
            }
          }
        });
      }
    }

    // Chart: Residuals
    if (!this.charts.residuals) {
      const ctx = document.getElementById('chart-residuals');
      if (ctx) {
        this.charts.residuals = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: ['-3 a -2', '-2 a -1', '-1 a 0', '0 a 1', '1 a 2', '2 a 3'],
            datasets: [{
              label: 'Frequência',
              data: [8, 25, 40, 38, 22, 7],
              backgroundColor: this.theme.c1,
              borderRadius: 4
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: { labels: { color: this.theme.text } }
            },
            scales: {
              x: {
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid, drawBorder: false }
              },
              y: {
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              }
            }
          }
        });
      }
    }

    // Populate metrics
    const reg = this.data.regression;
    document.getElementById('reg-rmse').textContent = (reg.rmse || 2.34).toFixed(2);
    document.getElementById('reg-mae').textContent = (reg.mae || 1.67).toFixed(2);
    document.getElementById('reg-r2').textContent = (reg.r2 * 100).toFixed(1) + '%';
  }

  initClusteringCharts() {
    // Chart: Cluster Distribution
    if (!this.charts.clusterDistribution) {
      const ctx = document.getElementById('chart-cluster-distribution');
      if (ctx) {
        this.charts.clusterDistribution = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'],
            datasets: [{
              label: 'Quantidade de Vagas',
              data: [1050, 980, 1200, 1150, 854],
              backgroundColor: [this.theme.c1, this.theme.c2, this.theme.success, this.theme.c4, this.theme.c5],
              borderRadius: 6
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            indexAxis: 'y',
            plugins: {
              legend: { labels: { color: this.theme.text } }
            },
            scales: {
              x: {
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              },
              y: {
                ticks: { color: this.theme.muted }
              }
            }
          }
        });
      }
    }

    // Chart: Cluster Sizes
    if (!this.charts.clusterSizes) {
      const ctx = document.getElementById('chart-cluster-sizes');
      if (ctx) {
        this.charts.clusterSizes = new Chart(ctx, {
          type: 'radar',
          data: {
            labels: ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'],
            datasets: [{
              label: 'Tamanho Relativo',
              data: [1050, 980, 1200, 1150, 854],
              borderColor: this.theme.c1,
              backgroundColor: this.withAlpha(this.theme.c1, 0.16),
              fill: true
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: { labels: { color: this.theme.text } }
            },
            scales: {
              r: {
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              }
            }
          }
        });
      }
    }

    // Populate metrics
    const clust = this.data.clustering;
    document.getElementById('clust-k').textContent = clust.k || 5;
    document.getElementById('clust-silhouette').textContent = (clust.silhouette).toFixed(2);
    document.getElementById('clust-samples').textContent = (clust.samples || 5234).toLocaleString('pt-BR');
  }

  initRecommendationCharts() {
    // Chart: Relevance Trend
    if (!this.charts.relevanceTrend) {
      const ctx = document.getElementById('chart-relevance-trend');
      if (ctx) {
        this.charts.relevanceTrend = new Chart(ctx, {
          type: 'line',
          data: {
            labels: ['Dia 1', 'Dia 2', 'Dia 3', 'Dia 4', 'Dia 5', 'Dia 6', 'Dia 7'],
            datasets: [{
              label: 'Taxa de Relevância',
              data: [0.78, 0.80, 0.82, 0.81, 0.83, 0.84, 0.84],
              borderColor: this.theme.c1,
              backgroundColor: this.withAlpha(this.theme.c1, 0.16),
              fill: true,
              tension: 0.4,
              borderWidth: 2
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: { labels: { color: this.theme.text } }
            },
            scales: {
              y: {
                min: 0.7,
                max: 1,
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              },
              x: {
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              }
            }
          }
        });
      }
    }

    // Chart: Recommendation Scores
    if (!this.charts.recommendationScores) {
      const ctx = document.getElementById('chart-recommendation-scores');
      if (ctx) {
        this.charts.recommendationScores = new Chart(ctx, {
          type: 'bar',
          data: {
            labels: ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'],
            datasets: [{
              label: 'Recomendações',
              data: [45, 120, 380, 920, 2769],
              backgroundColor: this.theme.c2,
              borderRadius: 4
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: { labels: { color: this.theme.text } }
            },
            scales: {
              x: {
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid, drawBorder: false }
              },
              y: {
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              }
            }
          }
        });
      }
    }

    // Chart: Top Recommended Skills
    if (!this.charts.topRecommendedSkills) {
      const ctx = document.getElementById('chart-top-recommended-skills');
      if (ctx) {
        this.charts.topRecommendedSkills = new Chart(ctx, {
          type: 'barh',
          data: {
            labels: ['Python', 'JavaScript', 'SQL', 'React', 'Data Science', 'AWS', 'Docker', 'Kubernetes', 'TypeScript', 'Git'],
            datasets: [{
              label: 'Frequência de Recomendação',
              data: [1250, 1100, 980, 850, 780, 650, 580, 450, 420, 380],
              backgroundColor: this.theme.success,
              borderRadius: 4
            }]
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
              legend: { display: false }
            },
            scales: {
              x: {
                ticks: { color: this.theme.muted },
                grid: { color: this.theme.grid }
              },
              y: {
                ticks: { color: this.theme.muted }
              }
            }
          }
        });
      }
    }

    // Populate metrics
    const rec = this.data.recommendation;
    document.getElementById('rec-hit-rate').textContent = (rec.hitRate * 100).toFixed(1) + '%';
    document.getElementById('rec-latency').textContent = rec.latency || 234;
    document.getElementById('rec-coverage').textContent = (rec.coverage * 100).toFixed(1) + '%';
  }

  renderEndpointsTable() {
    const tbody = document.getElementById('table-endpoints');
    const now = new Date();
    const formatTime = (minutesAgo) => {
      const date = new Date(now.getTime() - minutesAgo * 60000);
      return date.toLocaleString('pt-BR', { hour12: false });
    };
    const endpoints = [
      { name: 'Classificação', url: '/ml/classification/predict', status: 'online', time: formatTime(3) },
      { name: 'Regressão', url: '/ml/regression/predict', status: 'online', time: formatTime(7) },
      { name: 'Clusterização', url: '/ml/clustering/predict', status: 'online', time: formatTime(18) },
      { name: 'Recomendação', url: '/ml/recommendation/predict', status: 'online', time: formatTime(4) }
    ];

    tbody.innerHTML = endpoints.map(ep => `
      <tr>
        <td><strong>${ep.name}</strong></td>
        <td><code class="code-pill">${ep.url}</code></td>
        <td><span class="badge badge-success">${ep.status}</span></td>
        <td>${ep.time}</td>
        <td><span class="text-success">${120 + Math.floor(Math.random() * 80)} ms</span></td>
      </tr>
    `).join('');
  }

  updateContextUrl() {
    const label = document.getElementById('context-api-url');
    if (label) label.textContent = this.apiUrl;
  }

  async predictClassification() {
    const input = document.getElementById('clf-input').value;
    if (!input) {
      alert('Por favor, insira um texto de habilidade');
      return;
    }

    try {
      if (this.mockMode) {
        // Mock prediction
        const result = Math.random() > 0.5;
        this.showClassificationResult(result);
      } else {
        const response = await this.apiCall('/ml/classification/predict', 'POST', { text: input });
        this.showClassificationResult(response.prediction);
      }
    } catch (error) {
      alert('Erro ao fazer previsão: ' + error.message);
    }
  }

  showClassificationResult(prediction) {
    const resultDiv = document.getElementById('clf-result');
    const resultText = document.getElementById('clf-result-text');

    const isPositive = typeof prediction === 'boolean' ? prediction : prediction > 0.5;
    resultText.textContent = isPositive
      ? 'Habilidade detectada na vaga'
      : 'Habilidade não detectada';

    resultDiv.style.display = 'block';
  }

  async predictRegression() {
    const input = document.getElementById('reg-input').value;
    if (!input) {
      alert('Por favor, insira uma descrição de vaga');
      return;
    }

    try {
      if (this.mockMode) {
        // Mock prediction
        const result = Math.floor(Math.random() * 10) + 3;
        this.showRegressionResult(result);
      } else {
        const response = await this.apiCall('/ml/regression/predict', 'POST', { text: input });
        this.showRegressionResult(response.prediction);
      }
    } catch (error) {
      alert('Erro ao fazer previsão: ' + error.message);
    }
  }

  showRegressionResult(prediction) {
    const resultDiv = document.getElementById('reg-result');
    const resultText = document.getElementById('reg-result-text');

    resultText.textContent = `${prediction.toFixed(0)} habilidades`;

    resultDiv.style.display = 'block';
  }

  async predictRecommendation() {
    const jobId = document.getElementById('rec-input').value;
    const count = parseInt(document.getElementById('rec-count').value);

    if (!jobId) {
      alert('Por favor, insira um ID de vaga');
      return;
    }

    try {
      if (this.mockMode) {
        // Mock recommendations
        const jobs = [
          { id: 'JOB001', title: 'Senior Data Scientist', similarity: 0.95 },
          { id: 'JOB002', title: 'Machine Learning Engineer', similarity: 0.92 },
          { id: 'JOB003', title: 'Data Engineer', similarity: 0.88 },
          { id: 'JOB004', title: 'AI Researcher', similarity: 0.85 },
          { id: 'JOB005', title: 'Analytics Developer', similarity: 0.82 }
        ];
        this.showRecommendations(jobs.slice(0, count));
      } else {
        const response = await this.apiCall('/ml/recommendation/predict', 'POST', {
          job_id: jobId,
          top_k: count
        });
        this.showRecommendations(response.recommendations);
      }
    } catch (error) {
      alert('Erro ao fazer recomendação: ' + error.message);
    }
  }

  showRecommendations(jobs) {
    const resultDiv = document.getElementById('rec-result');
    const resultList = document.getElementById('rec-result-list');

    resultList.innerHTML = jobs.map(job => `
      <li>
        <span class="list-label">${job.title || job.name}</span>
        <span class="list-value list-pill">
          ${(job.similarity * 100).toFixed(1)}% similar
        </span>
      </li>
    `).join('');

    resultDiv.style.display = 'block';
  }

  refreshDashboard() {
    const btn = document.getElementById('btn-refresh');
    btn.disabled = true;
    btn.textContent = 'Atualizando...';

    setTimeout(() => {
      // Destroy and recreate charts
      Object.values(this.charts).forEach(chart => {
        if (chart && chart.destroy) chart.destroy();
      });
      this.charts = {};

      // Reload data
      this.checkAPIConnection();
      this.loadMockData();

      btn.disabled = false;
      btn.textContent = 'Atualizar';
    }, 1500);
  }

  exportData() {
    const exportData = {
      timestamp: new Date().toISOString(),
      api_url: this.apiUrl,
      data: this.data,
      charts: Object.keys(this.charts)
    };

    const json = JSON.stringify(exportData, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ml-dashboard-export-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.dashboard = new MLDashboard();
});


