@echo off
setlocal
echo "Building multi-arch images and pushing to AWS ECR..."

REM Load environment variables from .env file
cd ..
for /f "tokens=1,2 delims==" %%a in (.env) do set %%a=%%b

REM Construct full ECR image URI
set ECR_URI=%AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION_NAME%.amazonaws.com/%ECR_REPO_NAME%

REM Make sure you're logged in: aws sso login --profile <profile>
REM Login to AWS ECR
aws --profile %AWS_PROFILE_ADMIN% ecr get-login-password --region %AWS_REGION_NAME% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION_NAME%.amazonaws.com

REM Ensure repository exists
aws --profile %AWS_PROFILE_ADMIN% ecr describe-repositories --repository-names %ECR_REPO_NAME% --region %AWS_REGION_NAME% >nul 2>&1
if errorlevel 1 (
    echo "ECR repo not found. Creating it..."
    aws --profile %AWS_PROFILE_ADMIN% ecr create-repository --repository-name %ECR_REPO_NAME% --region %AWS_REGION_NAME%
)

REM Remove old local images
docker rmi "%ECR_URI%.base:latest" -f
docker rmi "%ECR_URI%:latest" -f

REM Build and push backend image
docker buildx build --platform linux/amd64,linux/arm64 ^
  -f "Dockerfile" ^
  -t "%ECR_URI%:latest" ^
  --push .

echo "Multi-arch build and push completed successfully."
pause
