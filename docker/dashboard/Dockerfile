# docker/dashboard/Dockerfile
# Build stage
FROM node:18-alpine as build

WORKDIR /app
COPY services/dashboard/package*.json ./
RUN npm ci --only=production

COPY services/dashboard/ .
RUN npm run build

# Production stage
FROM nginx:alpine

# Copie des fichiers build
COPY --from=build /app/dist /usr/share/nginx/html

# Configuration Nginx
COPY docker/dashboard/nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]