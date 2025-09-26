
@echo off
echo "Starting the backend server..."

cd ..
docker-compose -f "docker-compose.yml" -p "backend" up -d

echo "Program completed. If an error arose, see the message above."
pause