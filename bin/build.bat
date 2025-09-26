@echo off
setlocal
echo "Building images locally for x86_64 architecture..."

cd ..

docker build -f "Dockerfile" -t "hyper-rag-backend/backend:latest" .

docker image prune -f

echo "Local build completed successfully."

pause
