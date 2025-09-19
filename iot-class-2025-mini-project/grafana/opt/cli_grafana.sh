#!/bin/bash
# ⚡ Create Grafana Org + User for 35 students
# ✅ Each user is Admin of their own Org
# ✅ Using Basic Auth (admin:admin)

GRAFANA_URL="http://localhost:3000"
ADMIN_USER="admin"
ADMIN_PASS="admin"

students=(
6310301030
6410301001
6410301032
6510301002
6510301003
6510301004
6510301007
6510301010
6510301011
6510301012
6510301013
6510301014
6510301015
6510301016
6510301017
6510301018
6510301019
6510301023
6510301024
6510301025
6510301026
6510301029
6510301032
6510301033
6510301035
6510301041
6510301042
6510301044
6510301046
6510301047
6510301048
6520301001
6520301002
)

for sid in "${students[@]}"; do
  ORG_NAME="org-${sid}"
  USER_LOGIN="${sid}"
  USER_PASSWORD="pw${sid}"
  USER_EMAIL="${sid}@example.com"

  echo "=== Processing $sid ==="

  # 1. Create Org
  curl -s -X POST \
       -u "$ADMIN_USER:$ADMIN_PASS" \
       -H "Content-Type: application/json" \
       -d "{\"name\":\"$ORG_NAME\"}" \
       "$GRAFANA_URL/api/orgs" || echo "⚠️ Org $ORG_NAME may already exist"

  # 2. Get Org ID
  ORG_ID=$(curl -s -u "$ADMIN_USER:$ADMIN_PASS" \
       "$GRAFANA_URL/api/orgs/name/$ORG_NAME" | jq -r '.id')

  # 3. Create User
  curl -s -X POST \
       -u "$ADMIN_USER:$ADMIN_PASS" \
       -H "Content-Type: application/json" \
       -d "{\"name\":\"$USER_LOGIN\",\"login\":\"$USER_LOGIN\",\"email\":\"$USER_EMAIL\",\"password\":\"$USER_PASSWORD\"}" \
       "$GRAFANA_URL/api/admin/users" || echo "⚠️ User $USER_LOGIN may already exist"

  # 4. Add user to Org as Admin
  curl -s -X POST \
       -u "$ADMIN_USER:$ADMIN_PASS" \
       -H "Content-Type: application/json" \
       -d "{\"loginOrEmail\":\"$USER_LOGIN\",\"role\":\"Admin\"}" \
       "$GRAFANA_URL/api/orgs/${ORG_ID}/users"

  echo "✅ Created Grafana org=$ORG_NAME, user=$USER_LOGIN (Admin)"
done

echo "🎉 All Grafana users created as Admin of their own Org!"
