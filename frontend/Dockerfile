# =============================================================================
# Production Dockerfile for React Frontend
# Multi-stage build: Node.js for building, nginx for serving
# =============================================================================

# -----------------------------------------------------------------------------
# Build stage: Install dependencies and build the React app
# -----------------------------------------------------------------------------
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files first for better layer caching
COPY package.json package-lock.json* ./

# Install dependencies
# Use ci for reproducible builds when lock file exists, otherwise install
RUN if [ -f package-lock.json ]; then \
        npm ci --ignore-scripts; \
    else \
        npm install --ignore-scripts; \
    fi

# Copy source code
COPY . .

# Build the application
# VITE_API_BASE_URL is empty because nginx proxies /api to backend
ENV VITE_API_BASE_URL=""
RUN npm run build

# -----------------------------------------------------------------------------
# Production stage: Serve with nginx
# -----------------------------------------------------------------------------
FROM nginx:1.25-alpine AS production

# Security: Remove default nginx user and use a dedicated non-root user
# Note: nginx master process needs to start as root, but workers run as nginx user

# Remove default nginx configuration
RUN rm -rf /etc/nginx/conf.d/default.conf /usr/share/nginx/html/*

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Set proper permissions
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html && \
    # Create nginx cache directories with proper permissions
    mkdir -p /var/cache/nginx/client_temp \
             /var/cache/nginx/proxy_temp \
             /var/cache/nginx/fastcgi_temp \
             /var/cache/nginx/uwsgi_temp \
             /var/cache/nginx/scgi_temp && \
    chown -R nginx:nginx /var/cache/nginx && \
    # Create pid directory
    mkdir -p /var/run/nginx && \
    chown -R nginx:nginx /var/run/nginx

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

# Start nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
