-- These CREATE TABLE statements are placeholders when operating directly inside
-- DuckDB.  The Python feature pipeline loads dataframes into DuckDB temporary
-- tables so these statements are optional in tests.
CREATE TABLE IF NOT EXISTS games(
    game_pk BIGINT,
    date DATE,
    season INTEGER,
    home_id INTEGER,
    away_id INTEGER,
    venue VARCHAR,
    doubleheader_flag BOOLEAN,
    day_night VARCHAR,
    temperature DOUBLE
);
CREATE TABLE IF NOT EXISTS pbp(
    game_pk BIGINT,
    inning INTEGER,
    top_bottom VARCHAR,
    half_inning VARCHAR,
    at_bat_index INTEGER,
    batter_id BIGINT,
    pitcher_id BIGINT,
    event VARCHAR,
    description VARCHAR,
    runs_scored INTEGER,
    outs_before INTEGER,
    before_base_state STRUCT(first INTEGER, second INTEGER, third INTEGER),
    rbi INTEGER
);
CREATE TABLE IF NOT EXISTS box_team(
    game_pk BIGINT,
    team_id INTEGER,
    is_home BOOLEAN,
    R INTEGER,
    H INTEGER,
    BB INTEGER,
    SO INTEGER,
    HR INTEGER,
    "2B" INTEGER,
    "3B" INTEGER,
    SB INTEGER,
    CS INTEGER,
    E INTEGER,
    LOB INTEGER
);

CREATE VIEW IF NOT EXISTS team_games AS
WITH opp AS (
    SELECT
        game_pk,
        team_id,
        is_home,
        CASE WHEN is_home THEN 'home' ELSE 'away' END AS side,
        R
    FROM box_team
)
SELECT
    g.game_pk,
    g.date,
    g.season,
    g.day_night,
    g.doubleheader_flag,
    g.temperature,
    g.venue,
    t.team_id,
    t.is_home,
    CASE WHEN t.is_home THEN g.home_id ELSE g.away_id END AS team_lookup,
    CASE WHEN t.is_home THEN g.away_id ELSE g.home_id END AS opponent_id,
    CASE WHEN t.is_home THEN g.home_id ELSE g.away_id END AS primary_team,
    t.R AS team_runs,
    o.R AS opponent_runs,
    (t.R - o.R) AS run_diff,
    CASE WHEN t.R > o.R THEN 1 ELSE 0 END AS home_win,
    CASE WHEN t.R > o.R THEN 'win' ELSE 'loss' END AS team_result
FROM games g
JOIN box_team t ON t.game_pk = g.game_pk
JOIN box_team o ON o.game_pk = g.game_pk AND o.is_home <> t.is_home;

CREATE VIEW IF NOT EXISTS feature_scored_first AS
WITH scoring_half AS (
    SELECT
        p.game_pk,
        CASE WHEN p.top_bottom = 'top' THEN g.away_id ELSE g.home_id END AS team_id,
        ROW_NUMBER() OVER (PARTITION BY p.game_pk ORDER BY p.inning, p.at_bat_index) AS seq
    FROM pbp p
    JOIN games g USING(game_pk)
    WHERE p.runs_scored > 0
)
SELECT
    tg.game_pk,
    tg.team_id,
    CASE WHEN MIN(CASE WHEN scoring_half.team_id = tg.team_lookup THEN 1 ELSE 0 END) FILTER (WHERE scoring_half.seq = 1) = 1 THEN 1 ELSE 0 END AS scored_first
FROM team_games tg
LEFT JOIN scoring_half ON scoring_half.game_pk = tg.game_pk AND scoring_half.seq = 1
GROUP BY 1,2;

CREATE VIEW IF NOT EXISTS feature_big_inning AS
WITH inning_runs AS (
    SELECT
        p.game_pk,
        CASE WHEN p.top_bottom = 'top' THEN g.away_id ELSE g.home_id END AS team_id,
        p.inning,
        SUM(p.runs_scored) AS runs_in_inning
    FROM pbp p
    JOIN games g USING(game_pk)
    GROUP BY 1,2,3
)
SELECT
    tg.game_pk,
    tg.team_id,
    CASE WHEN MAX(CASE WHEN runs_in_inning >= 3 THEN 1 ELSE 0 END) = 1 THEN 1 ELSE 0 END AS big_inning
FROM team_games tg
LEFT JOIN inning_runs ir ON ir.game_pk = tg.game_pk AND ir.team_id = tg.team_lookup
GROUP BY 1,2;

CREATE VIEW IF NOT EXISTS feature_answered_back AS
WITH scoring AS (
    SELECT
        p.game_pk,
        p.inning,
        p.top_bottom,
        CASE WHEN p.top_bottom = 'top' THEN g.away_id ELSE g.home_id END AS team_id,
        SUM(p.runs_scored) AS runs
    FROM pbp p
    JOIN games g USING(game_pk)
    WHERE p.runs_scored > 0
    GROUP BY 1,2,3,4
),
ordered AS (
    SELECT
        *,
        LEAD(team_id) OVER (PARTITION BY game_pk ORDER BY inning, top_bottom) AS next_team,
        LEAD(runs) OVER (PARTITION BY game_pk ORDER BY inning, top_bottom) AS next_runs
    FROM scoring
)
SELECT
    tg.game_pk,
    tg.team_id,
    CASE WHEN SUM(CASE WHEN ordered.team_id <> ordered.next_team AND ordered.next_team = tg.team_lookup AND ordered.next_runs > 0 THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END AS answered_back_next_half
FROM team_games tg
LEFT JOIN ordered ON ordered.game_pk = tg.game_pk
GROUP BY 1,2;

CREATE VIEW IF NOT EXISTS feature_shutdown_after AS
WITH scoring_half AS (
    SELECT
        p.game_pk,
        p.inning,
        p.top_bottom,
        CASE WHEN p.top_bottom = 'top' THEN g.away_id ELSE g.home_id END AS team_id
    FROM pbp p
    JOIN games g USING(game_pk)
    WHERE p.runs_scored > 0
),
next_half AS (
    SELECT
        s.game_pk,
        s.team_id,
        LEAD(s.inning) OVER (PARTITION BY s.game_pk ORDER BY s.inning, s.top_bottom) AS next_inning,
        LEAD(s.top_bottom) OVER (PARTITION BY s.game_pk ORDER BY s.inning, s.top_bottom) AS next_half
    FROM scoring_half s
)
SELECT
    tg.game_pk,
    tg.team_id,
    CASE
        WHEN COUNT(next_half.next_inning) = 0 THEN NULL
        ELSE 1.0 - AVG(CASE WHEN COALESCE(opponent_runs.runs, 0) > 0 THEN 1 ELSE 0 END)
    END AS shutdown_inning_after_scoring
FROM team_games tg
LEFT JOIN next_half ON next_half.game_pk = tg.game_pk AND next_half.team_id = tg.team_lookup
LEFT JOIN (
    SELECT
        p.game_pk,
        p.inning,
        p.top_bottom,
        SUM(p.runs_scored) AS runs
    FROM pbp p
    GROUP BY 1,2,3
) AS opponent_runs
ON opponent_runs.game_pk = next_half.game_pk AND opponent_runs.inning = next_half.next_inning AND opponent_runs.top_bottom = next_half.next_half
GROUP BY 1,2;

CREATE VIEW IF NOT EXISTS feature_two_out_runs AS
WITH team_pbp AS (
    SELECT
        p.*,
        CASE WHEN p.top_bottom = 'top' THEN g.away_id ELSE g.home_id END AS batting_team
    FROM pbp p
    JOIN games g USING(game_pk)
)
SELECT
    tg.game_pk,
    tg.team_id,
    SUM(CASE WHEN tp.batting_team = tg.team_lookup AND tp.outs_before = 2 THEN tp.runs_scored ELSE 0 END) AS two_out_runs,
    SUM(CASE WHEN tp.batting_team = tg.team_lookup THEN tp.runs_scored ELSE 0 END) AS total_runs,
    CASE
        WHEN SUM(CASE WHEN tp.batting_team = tg.team_lookup THEN tp.runs_scored ELSE 0 END) = 0 THEN NULL
        ELSE SUM(CASE WHEN tp.batting_team = tg.team_lookup AND tp.outs_before = 2 THEN tp.runs_scored ELSE 0 END)::DOUBLE /
             SUM(CASE WHEN tp.batting_team = tg.team_lookup THEN tp.runs_scored ELSE 0 END)
    END AS two_out_run_share
FROM team_games tg
LEFT JOIN team_pbp tp ON tp.game_pk = tg.game_pk
GROUP BY 1,2;

CREATE VIEW IF NOT EXISTS feature_boxscore_diffs AS
SELECT
    tg.game_pk,
    tg.team_id,
    (t.H - opp.H) AS hits_diff,
    (t.BB - opp.BB) AS bb_diff,
    (t.SO - opp.SO) AS k_diff,
    (t.HR - opp.HR) AS hr_diff,
    ((t."2B" + t."3B") - (opp."2B" + opp."3B")) AS xbh_diff,
    (t.SB - opp.SB) AS sb_diff,
    (t.E - opp.E) AS e_diff,
    (t.LOB - opp.LOB) AS lob_diff
FROM team_games tg
JOIN box_team t ON t.game_pk = tg.game_pk AND t.team_id = tg.team_id
JOIN box_team opp ON opp.game_pk = tg.game_pk AND opp.is_home <> t.is_home;
