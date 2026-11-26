"""Ingredient and dense retrieval, pantry feature engineering, and proxy evaluations."""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import ndcg_score
from threadpoolctl import threadpool_limits

from catalog import IngredientNormalizer, SEED, STAPLES, OUT, family_key, ingredient_view, key, read_json, title_key, write_json
from encoder import encode, load_encoder, stable_ranking

BASELINE = {'core_coverage': 1.0, 'missing_essentials': -.11, 'weighted_coverage': .32, 'season_compatibility': .045}
FEATURES = ['core_coverage','missing_essentials','weighted_coverage','rarity','pantry_overlap','embedding_similarity','season_compatibility']
DISH_TYPE_BONUS = .24
TYPES = {
    'rice': r'\b(?:biryani|pulao|pilaf|palaw|palau|polao|khichuri|khichdi|rice|tehari)\b',
    'flatbread': r'\b(?:bread|roti|naan|paratha|chapati|kulcha|puri)\b',
    'stew': r'\b(?:curry|korma|masala|jhol|nihari|karahi|stew|bhuna)\b',
    'lentil': r'\b(?:dal|daal|haleem|sambar)\b',
    'snack': r'\b(?:pakora|samosa|cutlet|kebab|kabob|tikka|chaat|patties)\b',
    'sweet': r'\b(?:halwa|laddu|ladoo|kheer|payesh|barfi|burfi|sandesh|rasgulla|jalebi)\b',
}
TECHNIQUES = {
    'simmer': r'\b(?:simmer|stew|stewed|boil)\w*\b',
    'fry': r'\b(?:deep fry|shallow fry|fried|frying)\b',
    'grill': r'\b(?:grill|skewer|tandoor|charcoal)\w*\b',
    'steam': r'\b(?:steamed|steam|dum|layered)\b',
    'bake': r'\b(?:bake|oven|roast)\w*\b',
}
HAND_PANTRIES = [
    ('Winter vegetables and pulses', ['ফুলকপি','গাজর','palong shak','dal','alu','roshun'],'sheet'),
    ('Rice and mutton', ['basmati chal','khashi','পেঁয়াজ','আলু','ghee','elach'],'হেমন্ত'),
    ('Monsoon prawn pantry', ['chingri','pumpkin','কাঁচা মরিচ','coconut milk','roshun'],'barsha'),
    ('Paneer and greens', ['paneer','spinach','tomato','oil','পেঁয়াজ'],'winter'),
    ('Egg and chickpea cupboard', ['ডিম','ছোলা','besan','piyaz','jire'],'summer'),
    ('Tofu and fresh vegetables', ['tofu','tomato','eggplant','cucumber','ginger'],'basanta'),
    ('Sparse rice pantry', ['basmati rice','salt','onion'],None),
    ('Beef and mixed aromatics', ['gorur mangsho','পেঁয়াজ','রসুন','bay leaf','ধনে গুঁড়া'],'autumn'),
]

def recognized_types(title):
    return {name for name, pattern in TYPES.items() if re.search(pattern, title_key(title or ''))}

def infer_pasted_type(title, ingredients, steps):
    explicit = recognized_types(title)
    if explicit:
        return explicit
    has_rice = bool({'rice', 'basmati rice'} & set(ingredients)) or bool(re.search(r'\brice\b', steps, re.I))
    if has_rice and re.search(r'\b(?:layer|parboil|simmer|steam|pilaf|boil)\w*\b', steps, re.I):
        return {'rice'}
    return set()

def weighted_matrix(rows, words):
    pairs = [(i,words[name]) for i,names in enumerate(rows) for name in set(names) if name in words]
    coords = tuple(zip(*pairs)) if pairs else ([],[])
    return csr_matrix((np.ones(len(pairs),dtype=np.float32),coords),shape=(len(rows),len(words)))

class AmbiguousDish(ValueError):
    def __init__(self, candidates):
        super().__init__('Multiple dish families match. Supply a recipe ID or a more specific dish name.')
        self.candidates = candidates

class Retrieval:
    def __init__(self, directory=OUT, *, load_indexes=True, encoder_directory=None):
        self.directory = Path(directory)
        with (self.directory/'catalog.jsonl').open(encoding='utf-8') as stream:
            self.rows = [json.loads(line) for line in stream]
        self.id_map = {r['recipe_id']:i for i,r in enumerate(self.rows)}
        self.titles = defaultdict(list)
        for i,r in enumerate(self.rows):
            self.titles[r['title_key']].append(i)
        aliases = read_json(self.directory/'ingredient_aliases.json')
        self.normalizer = IngredientNormalizer(aliases['aliases'], aliases['known_dataset_names'])
        self.aliases = aliases['aliases']
        self.seasons = read_json(self.directory/'seasons.json')
        self.season_lookup = {key(x): k for k,v in self.seasons['seasons'].items() for x in [k,*v['aliases']]}
        self.ingredient_sets = [set(r['normalized_ingredients']) for r in self.rows]
        self.core_sets = [set(r['core']) for r in self.rows]
        self.words = {n:i for i,n in enumerate(sorted(set().union(*self.ingredient_sets)))}
        self.matrix = weighted_matrix(self.ingredient_sets,self.words)
        self.core = weighted_matrix(self.core_sets,self.words)
        train = [i for i,r in enumerate(self.rows) if r['split']=='train']
        frequency = np.asarray(self.matrix[train].sum(0)).ravel()
        self.idf = (1+np.log((1+len(train))/(1+frequency))).astype(np.float32)
        for name in STAPLES & self.words.keys():
            self.idf[self.words[name]] *= .20
        self.weighted_sizes = self.matrix @ self.idf
        self.core_sizes = np.asarray(self.core.sum(1)).ravel()
        self.split = np.array([r['split'] for r in self.rows])
        self.families = np.array([r['family'] for r in self.rows])
        self.groups = np.array([r['duplicate_group'] for r in self.rows])
        self.dish_types = [recognized_types(r['title']) for r in self.rows]
        self.embedding = {}
        self.encoder_directory = encoder_directory
        if load_indexes:
            for label, filename in (('frozen', 'frozen_embeddings.npz'), ('fine', 'recipe_embeddings.npz')):
                path = self.directory / filename
                if path.exists():
                    self.load_index(label, path)
        self.config = read_json(self.directory/'retrieval_config.json') if (self.directory/'retrieval_config.json').exists() else {}
        self.pantry_config = read_json(self.directory/'pantry_config.json') if (self.directory/'pantry_config.json').exists() else {}
        self.encoder = None
        self.ranker = None

    def load_index(self, label, path):
        with np.load(path) as archive:
            if 'recipe_ids' in archive and archive['recipe_ids'].tolist() != list(self.id_map):
                raise ValueError('Recipe index IDs do not match the catalog order.')
            block = {name: archive[name] for name in ('ingredients', 'steps')}
            if any(values.shape[0] != len(self.rows) for values in block.values()):
                raise ValueError('Recipe index rows do not match the catalog size.')
            self.embedding[label] = block

    def encode_text(self, text, model_name='fine'):
        if self.encoder is None or self.encoder[0] != model_name:
            if model_name == 'fine' and self.encoder_directory is not None:
                directory = self.encoder_directory()
            else:
                directory = self.directory / ('encoder' if model_name == 'fine' else 'frozen_encoder')
            if not directory.exists():
                return None
            self.encoder = (model_name, *load_encoder(directory))
        _, tokenizer, model = self.encoder
        return encode([text],tokenizer,model)[0]

    def season_key(self, text):
        if not text:
            return None
        canonical = self.season_lookup.get(key(text))
        if canonical is None:
            raise ValueError(f'Unknown season {text!r}. Choose from {sorted(self.seasons["seasons"])}.')
        return canonical

    def pantry_items(self, ingredients):
        if isinstance(ingredients,str):
            ingredients = re.split(r'[,;\n]',ingredients)
        raw = [str(x).strip() for x in ingredients if str(x).strip()]
        names = {self.normalizer(x) for x in raw}
        return names, sorted(x for x in raw if not self.normalizer.recognized(x) or self.normalizer(x) not in self.words)

    def mask(self, names):
        vector = np.zeros(len(self.words),dtype=np.float32)
        vector[[self.words[n] for n in names if n in self.words]] = 1
        return vector

    def jaccard(self, names):
        query = self.mask(names)*self.idf
        shared = self.matrix @ query
        return shared / np.maximum(self.weighted_sizes + query.sum() - shared, .0001)

    def resolve(self, query):
        token = query.strip()
        if token.upper() in self.id_map:
            return self.id_map[token.upper()], []
        q = title_key(token)
        direct = self.titles.get(q)
        if direct:
            if q=='kacchi biryani':
                representative=max(direct,key=lambda i:len(self.ingredient_sets[i]&{'mutton','basmati rice','potato','ghee','yogurt'}))
                return representative,[i for i in direct if i!=representative]
            return direct[0], direct[1:]
        canonical = family_key(q)
        candidates = [i for i,r in enumerate(self.rows) if r['family']==canonical]
        if candidates:
            return candidates[0], candidates[1:]
        similarity = sorted(((SequenceMatcher(a=q,b=t).ratio(),t) for t in self.titles),reverse=True)
        if not similarity or similarity[0][0] < .77:
            raise LookupError(f'No recipe title found for {query!r}')
        best = similarity[0][0]
        matches = [self.titles[t][0] for score,t in similarity if score >= max(.77,best-.035)]
        if len({self.families[i] for i in matches}) > 1:
            raise AmbiguousDish([self.summary(i) for i in matches[:8]])
        candidate = matches[0]
        return candidate, [i for i in self.titles[self.rows[candidate]['title_key']] if i!=candidate]

    def summary(self, idx):
        return {k:self.rows[idx][k] for k in ('recipe_id','title','cuisine','category','family')}

    def query_fields(self, query):
        if isinstance(query,dict):
            title, ingredients, steps = query.get('title'), query.get('ingredients',[]), query.get('steps',[])
            ingredients = self.pantry_items(ingredients)[0]
            steps = ' '.join(steps) if isinstance(steps,list) else str(steps)
            return title, ingredients, steps
        lines = [s.strip() for s in query.splitlines() if s.strip()]
        title, remaining, ingredients, directions = None, [], set(), []
        if lines and len(lines[0].split())<9 and not re.match(r'(ingredients|steps|directions|method|instructions)\b',lines[0],re.I):
            title,lines = lines[0],lines[1:]
        section = None
        normalized = sorted(self.normalizer.mapping.items(),key=lambda x:-len(x[0]))
        for line in lines:
            heading = re.match(r'^(ingredients?|steps?|directions|method|instructions)\s*:?',line,re.I)
            if heading:
                section = 'ingredients' if heading[1].lower().startswith('ingredient') else 'steps'
                line = line[heading.end():].strip()
            if not line:
                continue
            if section=='steps':
                directions.append(line)
            else:
                remaining.append(line)
                canonical = ' '+key(line)+' '
                matches = []
                for alias,name in normalized:
                    phrase = ' '+alias+' '
                    if phrase in canonical:
                        matches.append(name)
                        canonical = canonical.replace(phrase,' ')
                ingredients.update(matches)
                if section is None and len(line.split())>=6 and re.search(r'\b(add|mix|heat|bring|simmer|cook|chop|stir|fry|layer)\b',line,re.I):
                    directions.append(line)
        return title, ingredients, ' '.join(directions)

    def top(self, scores, n, allowed=None):
        chosen, group_seen, family_seen = [], set(), set()
        for i in stable_ranking(scores):
            if not np.isfinite(scores[i]) or (allowed is not None and not allowed[i]):
                continue
            if self.groups[i] in group_seen or self.families[i] in family_seen:
                continue
            group_seen.add(self.groups[i])
            family_seen.add(self.families[i])
            chosen.append(int(i))
            if len(chosen)==n:
                break
        return chosen

    def similarity_vectors(self, view, label, source_idx=None, ingredients=None, steps=None):
        block = self.embedding[label]
        signals = []
        if source_idx is not None:
            signals.append(block['ingredients'] @ block['ingredients'][source_idx])
            if view=='both':
                signals.append(block['steps'] @ block['steps'][source_idx])
        else:
            if ingredients:
                q = self.encode_text(ingredient_view(sorted(ingredients)),label)
                if q is not None:
                    signals.append(block['ingredients'] @ q)
            if view=='both' and steps:
                q = self.encode_text(steps,label)
                if q is not None:
                    signals.append(block['steps'] @ q)
        return np.mean(signals,axis=0) if signals else np.zeros(len(self.rows))

    def find(self, query, top_k):
        if top_k < 1:
            raise ValueError('top_k must be positive')
        source, variants, steps, title = None, [], '', None
        pasted = isinstance(query,dict) or '\n' in query or len(query.split())>=12
        if pasted:
            title, ingredients, steps = self.query_fields(query)
            if title:
                try:
                    source, variants = self.resolve(title)
                except AmbiguousDish as error:
                    return {'status':'ambiguous','choices':error.candidates,'results':[]}
                except LookupError:
                    pass
            if not ingredients and not steps:
                raise ValueError('Provide recognized ingredients or labeled cooking steps.')
        else:
            try:
                source,variants = self.resolve(query)
            except AmbiguousDish as error:
                return {'status':'ambiguous','choices':error.candidates,'results':[]}
            ingredients = self.ingredient_sets[source]
        use = self.config.get('production_model','fine' if 'fine' in self.embedding else 'ingredients')
        weight = self.config.get('embedding_weight',.65)
        scores = self.jaccard(ingredients)
        allowed = np.ones(len(self.rows),dtype=bool)
        if source is not None:
            allowed &= (self.families!=self.families[source]) & (self.groups!=self.groups[source])
        inferred_family = None
        if pasted and source is None:
            # Old Dhaka kacchi has recognizable raw-marinated mutton, rice layers and covered cooking.
            has_mutton = bool({'mutton','lamb'} & ingredients) or bool(re.search(r'\b(?:mutton|lamb)\b',steps,re.I))
            if (has_mutton and infer_pasted_type(title,ingredients,steps)=={'rice'}
                and re.search(r'\bmarinat\w*\b',steps,re.I) and re.search(r'\blayer\w*\b',steps,re.I)
                and re.search(r'\b(?:seal\w*|cover\w*|dum)\b',steps,re.I)):
                inferred_family = 'kacchi biryani'
                allowed &= self.families!=inferred_family
        if pasted and ingredients:
            overlap = self.matrix @ self.mask(ingredients)
            unweighted_jaccard = overlap/np.maximum(np.asarray(self.matrix.sum(1)).ravel()+len(ingredients)-overlap,1.)
            allowed &= unweighted_jaccard < .9
        if pasted and steps and use not in self.embedding and self.embedding:
            use = self.config.get('steps_query_model',next(iter(self.embedding)))
            weight = self.config.get('steps_query_weight',.35)
        if use in self.embedding:
            vectors = self.similarity_vectors('both',use,None if pasted else source,ingredients,steps)
            scores = weight * vectors + (1-weight)*scores if ingredients else vectors
        elif not ingredients:
            raise RuntimeError('Steps-only similarity requires the exported encoder.')
        query_types = (infer_pasted_type(title,ingredients,steps) if pasted else self.dish_types[source])
        boost = self.dish_type_boost(query_types)
        scores = scores + boost
        return {'status':'ok','source':self.summary(source) if source is not None else None,
            'same_title_or_family_variants':len(variants), 'variant_examples':[self.summary(i) for i in variants[:5]],
            'method':use,'query_dish_types':sorted(query_types),'inferred_excluded_family':inferred_family,
            'results':[dict(self.summary(i),score=float(scores[i]),
                        dish_type_matches=sorted(query_types & self.dish_types[i]),
                        dish_type_boost=float(boost[i]),
                        matched_ingredients=sorted(ingredients & self.ingredient_sets[i])) for i in self.top(scores,top_k,allowed)]}

    def dish_type_boost(self, query_types):
        return np.fromiter((DISH_TYPE_BONUS if query_types & candidate else 0. for candidate in self.dish_types),
                           dtype=np.float32,count=len(self.rows))

    def features(self, names, season, embedding=None):
        pantry = self.mask(names)
        core_shared = self.core @ pantry
        weighted = self.matrix @ (pantry*self.idf)
        seasonal = self.seasons['seasons'][season]['produce'] if season else []
        compatibility = np.minimum(self.matrix @ self.mask(seasonal),1)
        matches = self.matrix @ pantry
        return np.column_stack((core_shared/np.maximum(self.core_sizes,1),self.core_sizes-core_shared,
            weighted / np.maximum(self.weighted_sizes,.01),weighted/np.maximum(matches,1),
            matches/max(len(names),1),embedding if embedding is not None else np.zeros(len(self.rows)),
            compatibility)).astype(np.float32)

    def baseline(self, features):
        weights = {**BASELINE,'season_compatibility':self.seasons.get('boost',BASELINE['season_compatibility'])}
        return features @ np.array([weights.get(n,0.) for n in FEATURES],dtype=np.float32)

    def recommend(self, ingredients, season, top_k, selected=None):
        if top_k < 1:
            raise ValueError('top_k must be positive')
        names, unresolved = self.pantry_items(ingredients)
        if not names & self.words.keys():
            return {'season':self.season_key(season),'unresolved_ingredients':unresolved,'results':[]}
        season = self.season_key(season)
        selected = selected or self.pantry_config.get('production_model','weighted_baseline')
        dense = None
        if selected=='lambdarank_embedding' and 'fine' in self.embedding:
            vector = self.encode_text(ingredient_view(sorted(names)))
            dense = self.embedding['fine']['ingredients'] @ vector
        features = self.features(names,season,dense)
        seasonal_boost = features[:,-1]*self.seasons.get('boost',BASELINE['season_compatibility'])
        if selected.startswith('lambdarank'):
            import lightgbm as lgb
            if self.ranker is None or self.ranker[0]!=selected:
                self.ranker=(selected,lgb.Booster(model_file=str(self.directory/(selected+'.txt'))))
            indices = list(range(7)) if selected=='lambdarank_embedding' else [0,1,2,3,4,6]
            neutral = features.copy()
            neutral[:,-1] = 0
            scores = self.ranker[1].predict(neutral[:,indices],num_threads=2)+seasonal_boost
        else:
            scores = self.baseline(features)
        recipes = []
        for idx in self.top(scores,top_k,features[:,4]>0):
            missing = sorted(self.core_sets[idx]-names)
            recipes.append(dict(self.summary(idx),matched_ingredients=sorted(self.ingredient_sets[idx] & names),
                 missing_essentials=missing,seasonal_boost=float(seasonal_boost[idx]),
                 in_season_ingredients=sorted(set(self.seasons['seasons'][season]['produce']) & self.ingredient_sets[idx]) if season else [],
                 missing_in_season_ingredients=sorted((set(self.seasons['seasons'][season]['produce']) & self.ingredient_sets[idx])-names) if season else [],
                 core_coverage=float(features[idx,0]),score=float(scores[idx])))
        return {'season':season,'season_mapping_status':self.seasons['status'],'unresolved_ingredients':unresolved,
                'method':selected,'results':recipes}

def evaluate_similarity(engine, out):
    def tag(row):
        title = row['title_key']
        direction = ' '.join(s['text'].lower() for s in row['steps'])
        types = recognized_types(title)
        if not types:
            types = {'category:'+key(row['category'])}
        methods = {x for x,pat in TECHNIQUES.items() if re.search(pat,direction)}
        protein = {x['name'] for x in row['ingredients'] if x['category']=='Proteins'}
        dominant = protein or set(row['core'][:2])
        return types,methods,dominant
    tags = [tag(r) for r in engine.rows]
    def grade(i,j):
        a,b=tags[i],tags[j]
        return (int(bool(a[0] & b[0]))+int(bool(a[1] & b[1]) and bool(a[0] & b[0]))
                +int(bool(a[2] & b[2])))
    rng=np.random.default_rng(SEED)
    order = rng.permutation(len(engine.rows))
    pool, scores_rows, contrasts = [], [], []
    configs=[('ingredient_jaccard',None,0.)]
    for label in engine.embedding:
        for view in ('ingredients','both'):
            for weight in (.35,.65,1.):
                configs.append((f'{label}_{view}_{weight:.2f}',(label,view),weight))
    for split in ('validation','test'):
        distinct=set()
        query_idxs=[]
        for i in order:
            if engine.split[i]==split and tags[i][0] and engine.families[i] not in distinct:
                distinct.add(engine.families[i])
                query_idxs.append(i)
                if len(distinct)==24:
                    break
        if not query_idxs:
            continue
        all_scores=defaultdict(list)
        for i in query_idxs:
            allowed=(engine.families!=engine.families[i]) & (engine.groups!=engine.groups[i]) & (engine.split==split)
            ingredient=engine.jaccard(engine.ingredient_sets[i])
            systems={}
            for name, method, weight in configs:
                if method:
                    vector=engine.similarity_vectors(method[1],method[0],source_idx=i)
                    ranking=engine.top((1-weight)*ingredient+weight*vector,5,allowed)
                else:
                    ranking=engine.top(ingredient,5,allowed)
                systems[name]=ranking
            type_boost=engine.dish_type_boost(engine.dish_types[i])
            systems['ingredient_jaccard_with_type']=engine.top(ingredient+type_boost,5,allowed)
            if 'fine' in engine.embedding:
                vectors=engine.similarity_vectors('both','fine',source_idx=i)
                systems['fine_both_0.65_with_type']=engine.top(.35*ingredient+.65*vectors+type_boost,5,allowed)
            if 'frozen' in engine.embedding:
                vectors=engine.similarity_vectors('both','frozen',source_idx=i)
                systems['frozen_both_0.35_with_type']=engine.top(.65*ingredient+.35*vectors+type_boost,5,allowed)
            pooled=sorted(set().union(*systems.values()))
            judgments={j:grade(i,j) for j in pooled}
            ideal=sum((2**g-1)/np.log2(k+2) for k,g in enumerate(sorted(judgments.values(),reverse=True)[:5]))
            for name,ranking in systems.items():
                ndcg=sum((2**judgments[j]-1)/np.log2(k+2) for k,j in enumerate(ranking))/ideal if ideal else 0
                all_scores[name].append(ndcg)
            for j in pooled:
                pool.append({'split':split,'query_id':engine.rows[i]['recipe_id'],'query':engine.rows[i]['title'],
                    'candidate_id':engine.rows[j]['recipe_id'],'candidate':engine.rows[j]['title'],
                    'query_core':sorted(engine.core_sets[i]),'candidate_core':sorted(engine.core_sets[j]),
                    'query_steps':[s['text'] for s in engine.rows[i]['steps']],
                    'candidate_steps':[s['text'] for s in engine.rows[j]['steps']],
                    'proxy_grade':judgments[j],'independent_grade':None,
                    'rank_by_system':{name:ranks.index(j)+1 for name,ranks in systems.items() if j in ranks}})
            if split=='test' and 'fine_both_0.65' in systems:
                contrasts.append({'query':engine.rows[i]['title'], 'query_tags':[sorted(x) for x in tags[i]],
                    'jaccard_top':[(engine.rows[j]['title'],judgments[j],[sorted(x) for x in tags[j]]) for j in systems['ingredient_jaccard'][:2]],
                    'fine_with_steps_top':[(engine.rows[j]['title'],judgments[j],[sorted(x) for x in tags[j]]) for j in systems['fine_both_0.65'][:2]]})
        scores_rows.extend(dict(split=split,system=name,queries=len(vals),proxy_ndcg_at_5=float(np.mean(vals)))
                           for name,vals in all_scores.items())
    table=pd.DataFrame(scores_rows)
    table.to_csv(out/'similarity_metrics.csv',index=False)
    write_json(out/'similarity_candidate_pool.json',pool)
    write_json(out/'similarity_case_comparisons.json',contrasts)
    val=table[table.split.eq('validation') & ~table.system.str.endswith('_with_type')]
    best=val.sort_values('proxy_ndcg_at_5',ascending=False,kind='stable').iloc[0]['system'] if len(val) else 'ingredient_jaccard'
    # Select only full two-view options for the general API. Ingredient-only remains a controlled ablation.
    deployment=val[val.system.str.contains(r'_both_|ingredient_jaccard',regex=True)]
    selected=deployment.sort_values('proxy_ndcg_at_5',ascending=False,kind='stable').iloc[0]['system'] if len(deployment) else 'ingredient_jaccard'
    dense = val[val.system.str.contains(r'_both_',regex=True)]
    dense_selected = dense.sort_values('proxy_ndcg_at_5',ascending=False,kind='stable').iloc[0]['system'] if len(dense) else None
    config={'steps_query_model':dense_selected.split('_')[0] if dense_selected else None,
            'steps_query_weight':float(dense_selected.rsplit('_',1)[1]) if dense_selected else 0.,
            'production_model':'ingredients' if selected=='ingredient_jaccard' else selected.split('_')[0],
            'embedding_weight':0. if selected=='ingredient_jaccard' else float(selected.rsplit('_',1)[1]),
            'validation_winner_including_ablations':best,'selected_two_view_or_baseline':selected,
            'selection':'validation pooled automated proxy nDCG@5',
            'rubric':'One point shared CSV main ingredient, one point shared name-derived dish type (or CSV category for unrecognized titles), one point shared direction-derived technique conditional on same dish type.',
            'judgments':'automated heuristics; independent grades unavailable; inspect candidate pool',
            'dish_type_soft_boost':DISH_TYPE_BONUS,
            'dish_type_evaluation_note':'The extra with_type systems share a dish-type signal with the proxy rubric. Increased proxy scores do not establish independent relevance.'}
    write_json(out/'retrieval_config.json',config)
    print('SIMILARITY EVALUATION:',config,flush=True)
    return config

def train_pantry(engine,out):
    import lightgbm as lgb
    rng=np.random.default_rng(SEED)
    words=np.array(list(engine.words))
    available=list(engine.seasons['seasons'])
    samples={}
    for partition,group_count in [('train',550),('validation',100),('test',120)]:
        subsets=[]
        for src in rng.choice(np.flatnonzero(engine.split==partition),group_count,replace=False):
            core=sorted(engine.core_sets[src])
            if not core:
                continue
            selected=[x for x in core if rng.random()<rng.uniform(.48,.93)] or [core[0]]
            extras=sorted(engine.ingredient_sets[src]-set(core))
            selected.extend(x for x in extras if rng.random()<.48)
            selected.extend(rng.choice(words, size=int(rng.integers(0,4)),replace=False).tolist())
            alias_items=[str(rng.choice(engine.aliases.get(x,[x]))) if rng.random()<.4 else x for x in selected]
            pantry,_=engine.pantry_items(alias_items)
            season=available[int(rng.integers(len(available)))] if rng.random()<.7 else None
            subsets.append((int(src),pantry,season,alias_items))
        if 'fine' in engine.embedding and subsets:
            from encoder import load_encoder,encode
            if engine.encoder is None:
                engine.encoder=('fine',*load_encoder(out/'encoder'))
            _,tokenizer,model=engine.encoder
            dense=encode([ingredient_view(sorted(p)) for _,p,_,_ in subsets],tokenizer,model)
        else:
            dense=[None]*len(subsets)
        xs,ys,qs=[],[],[]
        for (src,pantry,season,raw),vector in zip(subsets,dense):
            embeddings=engine.embedding['fine']['ingredients']@vector if vector is not None else None
            feats=engine.features(pantry,season,embeddings)
            eligible=np.flatnonzero((engine.split==partition) & (engine.families!=engine.families[src]) & (engine.groups!=engine.groups[src]))
            baseline=engine.baseline(feats)
            ranked=eligible[stable_ranking(baseline[eligible])[:38]]
            background=rng.choice(eligible,min(14,len(eligible)),replace=False)
            candidates=np.unique(np.r_[ranked,background,src])
            # Weak labels reward the seed when sufficiently covered. Other recipes can also earn positives.
            coverage=feats[candidates,0]
            missing=feats[candidates,1]
            relevance=np.select([(coverage>=.90)&(missing==0), (coverage>=.65)&(missing<=2),coverage>=.35],[3,2,1],default=0)
            relevance[(candidates==src)&(coverage>=.65)]=3
            xs.append(feats[candidates])
            ys.append(relevance)
            qs.append((src,raw,season,candidates.tolist()))
        samples[partition]=(xs,ys,qs)
    train_x,train_y,_=samples['train']
    configurations={'lambdarank_no_embedding':[0,1,2,3,4,6]}
    if 'fine' in engine.embedding:
        configurations['lambdarank_embedding']=list(range(7))
    models={}
    for name,cols in configurations.items():
        model=lgb.LGBMRanker(objective='lambdarank',n_estimators=90,num_leaves=15,learning_rate=.055,
                              min_child_samples=24,n_jobs=4,random_state=SEED,verbosity=-1)
        model.fit(np.vstack(train_x)[:,cols],np.concatenate(train_y),group=[len(x) for x in train_x])
        model.booster_.save_model(str(out/(name+'.txt')))
        models[name]=(model,cols)
    metrics=[]
    for split in ('validation','test'):
        xs,ys,_=samples[split]
        for system in ['weighted_baseline',*models]:
            stats=[]
            for x,y in zip(xs,ys):
                if system=='weighted_baseline':
                    predicted=engine.baseline(x)
                else:
                    model,cols=models[system]
                    neutral=x.copy()
                    neutral[:,-1]=0
                    predicted=model.booster_.predict(neutral[:,cols],num_threads=2)+x[:,-1]*engine.seasons['boost']
                top=stable_ranking(predicted)[:5]
                stats.append([ndcg_score([2.**y-1],[predicted],k=5),x[top,0].mean(),x[top,1].mean()])
            mean=np.mean(stats,axis=0)
            metrics.append({'evaluation':split+'_weak','system':system,'n_queries':len(xs),
                           'proxy_ndcg_at_5':float(mean[0]),'core_coverage_at_5':float(mean[1]),'missing_core_at_5':float(mean[2])})
    samples_out=[]
    for system in ['weighted_baseline',*models]:
        summaries=[]
        for label,raw,season in HAND_PANTRIES:
            result=engine.recommend(raw,season,5,system)
            summaries.extend((r['core_coverage'],len(r['missing_essentials'])) for r in result['results'])
            samples_out.append({'scenario':label,'inputs':raw,'season':season,'ranker':system,'result':result})
        mean=np.mean(summaries,axis=0)
        metrics.append({'evaluation':'hand_authored_all_catalog','system':system,'n_queries':len(HAND_PANTRIES),
                       'proxy_ndcg_at_5':None,'core_coverage_at_5':float(mean[0]),'missing_core_at_5':float(mean[1])})
    table=pd.DataFrame(metrics)
    table.to_csv(out/'pantry_metrics.csv',index=False)
    write_json(out/'pantry_examples.json',samples_out)
    val=table[table.evaluation.eq('validation_weak')].sort_values('proxy_ndcg_at_5',ascending=False,kind='stable')
    selected=val.iloc[0].system
    # Require the chosen weak-label winner to transfer on both hand-authored aggregate measures.
    hand=table[table.evaluation.eq('hand_authored_all_catalog')].set_index('system')
    if (hand.loc[selected,'core_coverage_at_5'] < hand.loc['weighted_baseline','core_coverage_at_5'] or
        hand.loc[selected,'missing_core_at_5'] > hand.loc['weighted_baseline','missing_core_at_5']):
        selected='weighted_baseline'
    result={'production_model':selected,'validation_winner':val.iloc[0].system,'baseline_weights':BASELINE,'features':FEATURES,
            'weak_supervision':'Recipe-derived missing/core ingredients, extra ingredients, alias perturbations and same-partition candidates. Source gets grade 3 at >=65% core coverage. Other grades reflect core coverage/missing counts.',
            'selection':'Validation weak-label nDCG with a transfer check on both hand-authored measures.',
            'season':'Explicit soft bonus at inference; neither coverage nor missing essentials changes with season.',
            'hand_evaluation':'Hand-authored query inputs, automatic coverage/missing measurements, no human relevance labels.',
            'weak_query_counts':{split:len(data[0]) for split,data in samples.items()}}
    write_json(out/'pantry_config.json',result)
    print('PANTRY EVALUATION:',table.to_string(index=False),flush=True)
    return result

def run_retrieval(out=OUT):
    start=time.perf_counter()
    out=Path(out)
    engine=Retrieval(out)
    with threadpool_limits(limits=2, user_api='blas'):
        similarity=evaluate_similarity(engine,out)
        try:
            pantry=train_pantry(engine,out)
        except Exception as error:
            import traceback
            traceback.print_exc()
            pantry={'production_model':'weighted_baseline','baseline_weights':BASELINE,'training_error':str(error)}
            write_json(out/'pantry_config.json',pantry)
    write_json(out/'retrieval_timing.json',{'wall_seconds':time.perf_counter()-start})
    return similarity,pantry
