/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://smollm-backend:8001/:path*'
      }
    ]
  }
}

module.exports = nextConfig 