#!/bin/bash
# Google Cloud Storage Authentication Script

echo "============================================================"
echo "GOOGLE CLOUD STORAGE AUTHENTICATION"
echo "============================================================"

echo ""
echo "Step 1: Authenticating for gcloud CLI..."
echo "This will open a browser for authentication"
gcloud auth login

echo ""
echo "Step 2: Authenticating for API access..."
echo "This will open another browser for application-default credentials"
gcloud auth application-default login

echo ""
echo "Step 3: Setting project..."
gcloud config set project brb-traffic

echo ""
echo "============================================================"
echo "AUTHENTICATION COMPLETE"
echo "============================================================"
echo ""
echo "You can now run:"
echo "  python setup_gcs_and_download.py"
echo ""
