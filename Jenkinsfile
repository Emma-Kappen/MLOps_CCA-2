pipeline {
    agent any
    options { disableConcurrentBuilds() }
    triggers { cron('H 2 * * *') }
    environment {
        PROJECT_DIR = 'C:\\Users\\emkap\\Documents\\Projects\\MLOps_CCA-2'
        PY = 'venv\\Scripts\\python.exe'
    }
    stages {
        stage('Setup') {
            steps { dir(env.PROJECT_DIR) { bat '%PY% -m pip install -q -r requirements.txt' } }
        }
        stage('Data Pull & Prep') {
            steps { dir(env.PROJECT_DIR) { bat '%PY% src\\fetch_data.py' } }
        }
        stage('Model Retraining') {
            steps { dir(env.PROJECT_DIR) { bat '%PY% src\\train.py' } }
        }
        stage('Evaluation') {
            steps { dir(env.PROJECT_DIR) { bat '%PY% src\\evaluate.py' } }
        }
        stage('Deployment') {
            when { expression { fileExists("${env.PROJECT_DIR}\\artifacts\\eval_passed.flag") } }
            steps { dir(env.PROJECT_DIR) { bat '%PY% src\\deploy.py' } }
        }
    }
    post {
        always {
            dir(env.PROJECT_DIR) {
                bat(returnStatus: true, script: '%PY% src\\plot_metrics.py')
                archiveArtifacts artifacts: 'artifacts/metric_trend.png, artifacts/metrics_history.csv', allowEmptyArchive: true
            }
        }
        success { echo 'Model passed evaluation and was deployed to production/model.pkl' }
        failure { echo 'Pipeline failed — check logs (a failed Evaluation means the challenger was rejected and production is unchanged)' }
    }
}