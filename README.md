The following document states all adjustments needed to run the files in this repo on your own:

Programmatic_GenAI_Access.ipynb
-------------------------------
This notebook works on Google Colab. You'll need a HuggingFace token that can read the Llama model in the file, or whatever model you wish if you hope to customize. 
Store the token in your Google Colab Environment as a secret/key.

app.py
-------------------------------
This file requires the source of the flight summary .csv in Google Cloud Storage on line 126.

job.yaml
-------------------------------
This file requires the source of the flight summary .csv in Google Cloud Storage on line 19.
It also requires the source of the docker repo on line 16. Use the naming convention listed in the file to create the app repo on Cloud Shell to avoid trouble.

I deployed this project on Cloud Shell. Once these files all exist in your Cloud Shell (excluding the .ipynb), use the following commands in Cloud Shell to build it yourself:

(The below assumes all necessary APIs are enabled and the user is on the desired project)

1. gcloud artifacts repositories create llm-app-repo     --repository-format=docker     --location=us-central1     --description="LLM container images"

2. gcloud auth configure-docker us-central1-docker.pkg.dev

3. gcloud container clusters create flight-llm-cluster     --zone us-central1-a     --num-nodes 1     --machine-type n1-standard-4     --disk-size 50     --workload-pool=PROJECT ID HERE.svc.id.goog

4. gcloud container clusters get-credentials flight-llm-cluster --zone us-central1-a

5. kubectl create secret generic huggingface-token     --from-literal=token='YOUR TOKEN HERE'

6. Ensure you are in the project folder on cloud shell first!
   docker build -t us-central1-docker.pkg.dev/PROJECT_ID_HERE/llm-app-repo/flight-analysis-local:v1 .

7. docker push us-central1-docker.pkg.dev/PROJECT_ID_HERE/llm-app-repo/flight-analysis-local:v1 .

8. gcloud iam service-accounts create flight-analysis-sa     --display-name="Flight Analysis Service Account"

9. gcloud projects add-iam-policy-binding PROJECT_ID_HERE     --member="serviceAccount:flight-analysis-sa@PROJECT_ID_HERE.iam.gserviceaccount.com"     --role="roles/storage.objectViewer"

10. kubectl create serviceaccount flight-analysis-sa

11. gcloud iam service-accounts add-iam-policy-binding     flight-analysis-sa@PROJECT_ID_HERE.iam.gserviceaccount.com     --role roles/iam.workloadIdentityUser  \
   --member "serviceAccount:PROJECT_ID_HERE.svc.id.goog[default/flight-analysis-sa]"

12. kubectl annotate serviceaccount flight-analysis-sa     iam.gke.io/gcp-service-account=flight-analysis-sa@PROJECT_ID_HERE.iam.gserviceaccount.com

13. kubectl apply -f job.yaml

14. POD_NAME=$(kubectl get pods -l app=flight-analysis-local-llm -o jsonpath='{.items[0].metadata.name}')

15. kubectl logs -f $POD_NAME

The last two steps should help display the output of your Docker image in the terminal.

 
