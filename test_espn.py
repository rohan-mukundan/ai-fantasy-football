import subprocess, json

url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/BUF/roster"
result = subprocess.run(
    ["curl", "-v", "-s", "-L", "--max-time", "20",
     "-H", "User-Agent: Mozilla/5.0 Chrome/120.0.0.0",
     "-H", "Accept: application/json",
     url],
    capture_output=True, text=True
)
print("Return code:", result.returncode)
print("STDERR:", result.stderr[:1000])
print("STDOUT (first 500):", result.stdout[:500])
