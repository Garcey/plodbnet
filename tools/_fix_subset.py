from pathlib import Path
eb = Path("python/plo5bp/env_batched.py")
t = eb.read_text(encoding="utf-8")
old = '''                else:
                    obs_sub = encode_observation_batch(
                        bundle, cat_a, cat_b, self.config
                    )

        with record_function("step1a_unpack/post"):
            dones_sub = actors_sub == -1
'''
new = '''                else:
                    obs_sub = encode_observation_batch(
                        bundle, cat_a, cat_b, self.config
                    )
            if getattr(self, "_obs_mode", "full") == "minimal":
                obs_sub = project_obs_minimal(obs_sub)

        with record_function("step1a_unpack/post"):
            dones_sub = actors_sub == -1
'''
# also project rust subset path
if "observation_encoded_subset_batch" in t and "project_obs_minimal(obs_sub)" not in t.split("_refresh_subset")[1][:800]:
    pass
if old not in t:
    # show context
    i = t.find("def _refresh_subset")
    print(repr(t[i:i+1200][:800]))
    raise SystemExit("block not found")
t = t.replace(old, new)
# rust path: obs_sub from bundle also full width
old_r = '''            obs_sub = np.asarray(bundle["obs"], dtype=np.float32)
            actors_sub = np.asarray(bundle["actor"], dtype=np.int8)
        else:
'''
new_r = '''            obs_sub = np.asarray(bundle["obs"], dtype=np.float32)
            if getattr(self, "_obs_mode", "full") == "minimal":
                obs_sub = project_obs_minimal(obs_sub)
            actors_sub = np.asarray(bundle["actor"], dtype=np.int8)
        else:
'''
if old_r not in t:
    raise SystemExit("rust subset path not found")
t = t.replace(old_r, new_r)
eb.write_text(t, encoding="utf-8")
print("refresh_subset fixed")
