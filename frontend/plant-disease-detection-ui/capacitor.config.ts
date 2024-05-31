import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'io.ionic.starter',
  appName: 'plant-disease-detection-ui',
  webDir: 'www',

  server: {
    androidScheme: "http",
    cleartext: true,
    allowNavigation: [
      "http://ec2-18-197-167-218.eu-central-1.compute.amazonaws.com:5000/*"
    ]
  }
};

export default config;
